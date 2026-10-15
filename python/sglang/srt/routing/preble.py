# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Preble global scheduling adapted to the current SGLang transport.

Scheduling/cache semantics follow WukLab/preble at
1a35eae20c416f7062ca81775aab72890323fd97 (Apache-2.0).
Routing assignments and the virtual GPU cache are separate state.
"""

import heapq
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Sequence

import numpy as np

from sglang.srt.routing.preble_cache import LPRadixCache, TreeNode
from sglang.srt.routing.preble_cost import (
    SlidingWindowHistogram,
    TTFTWindowedOverloadedDetector,
    a6000_prefill_time,
)

logger = logging.getLogger(__name__)


@dataclass
class Assignment:
    leaf: TreeNode
    worker: int
    output_budget: int
    tokens: tuple[int, ...]
    finished: bool = False


class PrebleScheduler:
    """Reference E2, windowed group costs, virtual eviction and rebalancing.

    The frontend serializes access through the reference scheduler lock. Pool
    membership is fixed; an unavailable worker is excluded by the transport.
    """

    def __init__(
        self,
        num_workers: int,
        window_seconds: float = 180,
        prefill_time: Callable[[int, int], float] = a6000_prefill_time,
        initial_tpot: float = 0.15,
        enable_rebalancing: bool = True,
        enable_eviction: bool = False,
        max_tokens_gpu: Sequence[int] | None = None,
    ):
        if num_workers < 1 or window_seconds <= 0 or initial_tpot <= 0:
            raise ValueError("Worker count, window and TPOT must be positive")
        self.num_workers = self.num_gpus = num_workers
        self.gpu_allocations = {}
        self.counter = 0
        self.enable_eviction = enable_eviction
        self.per_gpu_load = {i: 0 for i in range(num_workers)}
        self.all_gpus = set(range(num_workers))
        self.lock = threading.Lock()
        self.avg_topt_per_gpu = [
            deque([initial_tpot], maxlen=200) for _ in range(num_workers)
        ]
        self.histogram = SlidingWindowHistogram(
            window_duration=timedelta(seconds=window_seconds),
            gpu_allocations=self.gpu_allocations,
            num_gpus=num_workers,
            avg_topt_per_gpu=self.avg_topt_per_gpu,
            prefill_time=prefill_time,
        )
        self.cache = LPRadixCache(
            histogram=self.histogram, num_gpus=num_workers, lock=self.lock
        )
        self.max_tokens_gpu = (
            list(max_tokens_gpu)
            if max_tokens_gpu is not None
            else [198516] * num_workers
        )
        if len(self.max_tokens_gpu) != num_workers or any(
            n <= 0 for n in self.max_tokens_gpu
        ):
            raise ValueError("One positive virtual KV capacity is required per worker")
        self.HIGH_LOAD_THRESHOLD = 1.5
        self.REBALANCING_CHAIN_LENGTH = 3
        self.overload_detector = TTFTWindowedOverloadedDetector(
            timedelta(seconds=window_seconds)
        )
        self.enable_rebalancing = enable_rebalancing
        self.migrations = 0
        self.replications = 0

    @property
    def scheduled(self):
        return self.counter

    @property
    def nodes(self):
        return self.cache.all_nodes

    @property
    def root(self):
        return self.cache.root_node

    def handle_split_nodes_gpu_allocations(self, split_nodes, gpu_allocations):
        for child_node, new_node in split_nodes.items():
            gpu_allocations[new_node] = gpu_allocations[child_node].copy()

    def handle_split_node_histogram(self, split_nodes: Dict[TreeNode, TreeNode]):
        for child, parent_node in split_nodes.items():
            if self.is_large_node(parent_node) and not self.is_large_node(
                child
            ):  # new node is parent is now larger
                self.histogram.rename_node(child, parent_node)
                parent_node.decode_length.extend(child.decode_length)
                child.decode_length = []
                for gpu in self.gpu_allocations.get(child, {}):
                    self.overload_detector.rename_node(child, parent_node, gpu)

            if self.is_large_node(child):
                for gpu in self.gpu_allocations.get(child, {}):
                    self.histogram.update_prefill_cost_per_node(child, gpu)

    def get_parent_gpu_allocation(self, node: TreeNode):
        if not node:
            return self.all_gpus
        if self.gpu_allocations.get(node):
            return self.gpu_allocations.get(node)
        return self.get_parent_gpu_allocation(node.parent)

    def update_gpu_allocation_for_parent(self, node: TreeNode, gpu_id):
        if not node:
            return
        self.gpu_allocations[node] = self.gpu_allocations.get(node, set()).union(gpu_id)
        self.update_gpu_allocation_for_parent(node.parent, gpu_id)

    def is_small_node(self, node: TreeNode):
        return not self.is_large_node(node)

    def is_large_node(self, node: TreeNode):
        return node.num_tokens > node.context_so_far

    def get_important_node(self, node: TreeNode):
        if self.is_large_node(node):
            return node
        return self.get_important_node(node.parent)

    def evict_callback(self, node: TreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        node.evicted_gpus.add(runtime_selected)
        node.cached_gpus.remove(runtime_selected)
        return len(node.value)

    def handle_eviction(self, runtime_selected):
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        assert self.cache.allocated_size(runtime_selected) >= 0
        if self.cache.allocated_size(runtime_selected) > current_max_tokens:
            num_tokens = (
                self.cache.allocated_size(runtime_selected) - current_max_tokens
            )
            self.cache.evict_with_runtime_id_without_removing(
                num_tokens,
                lambda node: self.evict_callback(node, runtime_selected),
                runtime_selected,
            )

    def virtual_evict_for_routing(self, leaf_node: TreeNode, runtime_selected: int):
        num_to_evict = leaf_node.num_tokens
        evicted_tree_nodes = self.cache.virtual_lru_eviction(
            num_to_evict, runtime_selected
        )
        eviction_cost = 0
        victim: TreeNode
        for victim in evicted_tree_nodes:
            eviction_cost += self.histogram.get_eviction_prefill_cost(
                victim, runtime_selected, self.is_large_node(victim)
            )
        return eviction_cost

    def calculate_min_load_cost(self, leaf_node, selected_gpus):
        histogram_mem_cost = self.histogram.current_allocation_per_gpu()
        costs = []
        for gpu_id in selected_gpus:
            cost = histogram_mem_cost[gpu_id]
            if self.enable_eviction:
                cost += self.virtual_evict_for_routing(leaf_node, gpu_id)
            costs.append(cost)
        gpu_selected = int(np.argmin(costs))
        # Preserve the reference's candidate-position result for source parity.
        return gpu_selected

    def handle_important_node_stealing(self, scheduled_idx):
        if sum(self.per_gpu_load.values()) < 50 * self.num_gpus:
            return
        allocation_cost_per_gpu = (
            self.histogram.current_allocation_per_gpu_with_atleast_min_load(2)
        )
        allocations_with_indices = [
            (gpu_id, allocation_cost_per_gpu[gpu_id])
            for gpu_id in range(len(allocation_cost_per_gpu))
        ]
        allocations_with_indices = list(
            sorted(allocations_with_indices, key=lambda x: -x[1])
        )
        self.handle_important_node_stealing_recursive(allocations_with_indices)

    def handle_important_node_stealing_recursive(self, allocation_cost_with_devices):
        if len(allocation_cost_with_devices) == 1:
            return
        larger_device, larger_allocation_cost = allocation_cost_with_devices[0]
        smaller_device, smaller_device_allocation_cost = allocation_cost_with_devices[
            -1
        ]  # Last element is the smallest

        if (
            larger_allocation_cost
            < self.HIGH_LOAD_THRESHOLD * smaller_device_allocation_cost
        ):
            return

        tpot = np.median(self.avg_topt_per_gpu[larger_device])
        node_cost_for_gpu = []
        all_rebalancing_cost = []
        for node, cost in self.histogram.histogram.items():
            if (
                larger_device in self.gpu_allocations.get(node)
                and self.is_large_node(node)
                and self.histogram.node_to_count[node] > 1
            ):
                rebalancing_cost = self.histogram.get_node_cost(
                    node, larger_device, tpot
                )
                heapq.heappush(node_cost_for_gpu, (rebalancing_cost, node))
                all_rebalancing_cost.append(rebalancing_cost)

        if len(node_cost_for_gpu) == 1:
            cost, node = node_cost_for_gpu[0]
            cost /= 2  # load is now split into two
            if smaller_device not in self.gpu_allocations[
                node
            ] and self.overload_detector.is_node_overloaded(node, larger_device):
                larger_allocation_cost -= cost
                smaller_device_allocation_cost += cost
                self.gpu_allocations[node].add(smaller_device)
                self.replications += 1
                self.overload_detector.delete_after_allocation(node, larger_device)
        else:
            steal_n = 0
            while node_cost_for_gpu:
                node: TreeNode
                cost, node = heapq.heappop(node_cost_for_gpu)

                assert self.is_large_node(node)

                if (
                    larger_allocation_cost - cost
                    < smaller_device_allocation_cost + cost
                ):
                    break
                larger_allocation_cost -= cost
                smaller_device_allocation_cost += cost
                self.gpu_allocations[node] = {smaller_device}

                self.histogram.migrate_node_cost(node, larger_device, smaller_device)
                self.update_children(node, smaller_device)
                steal_n += 1
                self.migrations += 1
                if (
                    larger_allocation_cost
                    < self.HIGH_LOAD_THRESHOLD * smaller_device_allocation_cost
                ):
                    break
            if steal_n != 0:
                logger.info(
                    f"Steal {steal_n} nodes from {larger_device} to {smaller_device}"
                )
        allocation_cost_with_devices[0] = (larger_device, larger_allocation_cost)
        allocation_cost_with_devices[-1] = (
            smaller_device,
            smaller_device_allocation_cost,
        )
        self.handle_important_node_stealing_recursive(allocation_cost_with_devices[1:])

    def update_children(self, node: TreeNode, gpu_id):
        for child in node.children.values():
            self.gpu_allocations[child] = {gpu_id}
            self.update_children(child, gpu_id)

    def select(self, tokens, output_budget, available=None):
        if not tokens or output_budget < 0:
            raise ValueError(
                "Nonempty input tokens and nonnegative output budget required"
            )
        available = set(range(self.num_gpus) if available is None else available)
        if not available or not available <= self.all_gpus:
            raise ValueError("No valid available worker")
        tokens = tuple(tokens)
        with self.lock:
            split_nodes = {}
            leaf_node = self.cache.insert(tokens, split_nodes=split_nodes)
            self.handle_split_nodes_gpu_allocations(split_nodes, self.gpu_allocations)
            self.handle_split_node_histogram(split_nodes)
            important_node = self.get_important_node(leaf_node)
            if leaf_node.num_tokens < leaf_node.context_so_far:
                candidates = self.get_parent_gpu_allocation(leaf_node) & available
                if not candidates:
                    candidates = available
                if len(candidates) > 1:
                    worker = self.calculate_min_load_cost(leaf_node, candidates)
                    if available != self.all_gpus:
                        worker = list(candidates)[worker]
                else:
                    worker = next(iter(candidates))
            else:
                worker = self.calculate_min_load_cost(leaf_node, sorted(available))
                if available != self.all_gpus:
                    # Cooldown filtering is a transport extension; remap its IDs.
                    worker = sorted(available)[worker]
            self.counter += 1
            self.update_gpu_allocation_for_parent(leaf_node, {worker})
            self.cache.update_allocated_size(leaf_node, worker)
            assert self.is_large_node(important_node)
            self.histogram.update(
                datetime.now(), important_node, leaf_node, worker, output_budget
            )
            self.per_gpu_load[worker] += 1
            if self.enable_eviction:
                self.handle_eviction(worker)
            if self.enable_rebalancing:
                if (
                    leaf_node.depth - important_node.depth
                    < self.REBALANCING_CHAIN_LENGTH
                ):
                    self.handle_important_node_stealing(worker)
            return Assignment(leaf_node, worker, output_budget, tokens)

    def finish(self, assignment, output_tokens=0, ttft=0, latency=0):
        """Keep reference feedback; release once on completion or cancellation."""
        with self.lock:
            if assignment.finished:
                return
            assignment.finished = True
            worker = assignment.worker
            leaf = self.cache.find_node(assignment.tokens)
            important_node = self.get_important_node(leaf)
            self.overload_detector.add_data_point(
                datetime.now(), important_node, worker, ttft
            )
            important_node = self.get_important_node(
                self.cache.find_node(assignment.tokens)
            )
            self.cache.remove_completed_input_ids(assignment.tokens, worker)
            if output_tokens > 1 and latency > ttft > 0:
                self.avg_topt_per_gpu[worker].append(
                    (latency - ttft) / (output_tokens - 1)
                )
            self.histogram.current_decode_lengths_per_gpu[
                worker
            ] -= assignment.output_budget
            if output_tokens:
                assert important_node in self.histogram.histogram
            self.histogram.per_node_total_decode_lengths[
                important_node
            ] -= assignment.output_budget
