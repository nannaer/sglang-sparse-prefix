# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Windowed cost and overload state from WukLab/preble (Apache-2.0).

Reference: global_scheduler_with_time.py and ttft_overload_detector.py at
1a35eae20c416f7062ca81775aab72890323fd97. A prefill function can be profiled
for the deployed model without changing the scheduling equations.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

from sglang.srt.routing.preble_cache import TreeNode


def a6000_prefill_time(tokens: int, context: int) -> float:
    """Original Mistral-7B/A6000 regression, in seconds; profile other hardware."""
    if tokens >= 384:
        linear = 0.10842571 * tokens + 4.209777054806409
    elif tokens >= 192:
        linear = -118 + 1.25 * tokens - 2.56e-3 * tokens**2
    else:
        linear = 22
    attention = 0.32 if context <= 1024 else 1.86e-4 * context + 0.159
    if context > 1024 and tokens <= 1024:
        attention /= 2
    quadratic = 0.0
    if tokens * context > 1024 * 1024:
        quadratic = 1.13e-3 * tokens + 1.75e-3 * context + 2.19e-6 * tokens * context
    return (linear + attention + quadratic) / 900


class SlidingWindowHistogram:
    def __init__(
        self,
        window_duration: timedelta,
        gpu_allocations,
        num_gpus=2,
        enable_miss_rate=True,
        avg_topt_per_gpu=None,
        prefill_time=None,
    ):
        self.prefill_time = prefill_time or a6000_prefill_time
        self.window_duration = window_duration
        self.gpu_allocations = gpu_allocations
        self.histogram = defaultdict(int)
        self.node_to_count = defaultdict(int)

        self.timestamps: List[Tuple[datetime, TreeNode, TreeNode]] = []
        self.num_gpus = num_gpus
        self.enable_miss_rate = enable_miss_rate
        self.prev_mis_rates = {}
        self.hit_tokens = defaultdict(int)
        self.prompt_tokens = defaultdict(int)
        self.decoding_size = defaultdict(int)
        self.avg_topt_per_gpu = avg_topt_per_gpu

        self.current_prefill_cost_per_gpu = [0 for i in range(self.num_gpus)]
        self.current_decode_lengths_per_gpu = [0 for i in range(self.num_gpus)]
        self.per_node_prefill_cost = defaultdict(int)
        self.per_node_total_decode_lengths = defaultdict(int)

    @property
    def current_allocation_cost_per_gpu(self):
        costs = []
        for i in range(self.num_gpus):
            topt = np.median(self.avg_topt_per_gpu[i])
            costs.append(
                self.current_prefill_cost_per_gpu[i]
                + self.current_decode_lengths_per_gpu[i] * topt
            )
        return costs

    def update(
        self,
        timestamp,
        node: TreeNode,
        leaf_node: TreeNode,
        runtime_idx,
        decoding_length,
    ):
        self.timestamps.append((timestamp, node, leaf_node))
        self.histogram[node] += 1 * leaf_node.context_length
        self.node_to_count[node] += 1
        self.decoding_size[node] = decoding_length
        self.per_node_total_decode_lengths[node] += decoding_length

        self.hit_tokens[node] += leaf_node.context_length - leaf_node.num_tokens
        self.prompt_tokens[node] += leaf_node.context_length

        self.prev_mis_rates[node] = 1 - (
            self.hit_tokens[node] / self.prompt_tokens[node]
        )
        self._remove_old_entries(timestamp)

        self.update_prefill_cost_per_node(node, runtime_idx)
        self.current_decode_lengths_per_gpu[runtime_idx] += decoding_length

    def update_prefill_cost_per_node(self, node, runtime_idx):
        if (
            runtime_idx not in self.gpu_allocations.get(node)
            or node not in self.prev_mis_rates
            or node not in self.node_to_count
        ):
            return  # only update gpu allocation for allocated nodes
        new_cost = self.get_prefill_cost(node)
        old_cost = self.per_node_prefill_cost[node]
        self.current_prefill_cost_per_gpu[runtime_idx] -= old_cost
        self.current_prefill_cost_per_gpu[runtime_idx] += new_cost
        self.per_node_prefill_cost[node] = new_cost

    def _remove_old_entries(self, current_timestamp):
        window_start = current_timestamp - self.window_duration
        while self.timestamps and self.timestamps[0][0] < window_start:
            timestamp, node, leaf_node = self.timestamps.pop(0)
            self.histogram[node] -= 1 * leaf_node.context_length
            self.node_to_count[node] -= 1
            self.hit_tokens[node] -= leaf_node.context_length - leaf_node.num_tokens
            self.prompt_tokens[node] -= leaf_node.context_length

            for gpu in range(self.num_gpus):
                self.update_prefill_cost_per_node(node, gpu)

            if self.histogram[node] <= 0:
                del self.histogram[node]
                del self.node_to_count[node]
                del self.prev_mis_rates[node]
                del self.hit_tokens[node]
                del self.prompt_tokens[node]
                del self.decoding_size[node]
                self.gpu_allocations[node] = (
                    set()
                )  # Reset the gpu allocation outside the time window

    def rename_node(self, old_node, new_node):
        if old_node in self.histogram:

            self.histogram[new_node] = self.histogram.pop(old_node)
            self.node_to_count[new_node] = self.node_to_count.pop(old_node)
            self.prev_mis_rates[new_node] = self.prev_mis_rates.pop(old_node)
            self.hit_tokens[new_node] = self.hit_tokens.pop(old_node)
            self.prompt_tokens[new_node] = self.prompt_tokens.pop(old_node)
            self.decoding_size[new_node] = self.decoding_size.pop(old_node)
            self.per_node_total_decode_lengths[new_node] = (
                self.per_node_total_decode_lengths.pop(old_node)
            )

            for gpu in range(self.num_gpus):
                self.per_node_prefill_cost[new_node] = self.per_node_prefill_cost[
                    old_node
                ]
                self.per_node_prefill_cost[old_node] = 0
                self.update_prefill_cost_per_node(new_node, gpu)

            timestamps = []
            for timestamp, important_node, leaf_node in self.timestamps:
                if important_node == old_node:
                    important_node = new_node
                timestamps.append((timestamp, important_node, leaf_node))
            self.timestamps = timestamps

    def query(self):
        return dict(self.histogram)

    def current_allocation_per_gpu(self):
        allocation = [0 for _ in range(self.num_gpus)]
        topts = []
        for i in range(self.num_gpus):
            topts.append(np.median(self.avg_topt_per_gpu[i]))
        node: TreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                allocation[gpu] += self.get_node_cost(node, gpu, topts[gpu])
        return allocation

    def current_allocation_per_gpu_with_atleast_min_load(self, min_load=2):
        allocation = [0 for _ in range(self.num_gpus)]
        topts = []
        for i in range(self.num_gpus):
            topts.append(np.median(self.avg_topt_per_gpu[i]))

        node: TreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                if self.node_to_count[node] < min_load:
                    continue
                allocation[gpu] += self.get_node_cost(node, gpu, topts[gpu])
        return allocation

    def get_prefill_cost(self, node: TreeNode):
        return (
            self.prev_mis_rates[node]
            * self.node_to_count[node]
            * self.prefill_time(node.num_tokens, node.context_length)
            / len(self.gpu_allocations.get(node))
        )  # potentionally divide by length of node.cached_gpus here

    def get_node_cost(self, node: TreeNode, gpu, tpot):
        prefill_cost = self.get_prefill_cost(node)
        output_len = self.decoding_size[node]
        if node.decode_length:
            output_len = np.median(node.decode_length)
        active_requests = node.ref_counter[gpu]
        decode_cost = active_requests * output_len * tpot
        return prefill_cost + decode_cost

    def get_eviction_prefill_cost(self, node: TreeNode, gpu, is_large_node: bool):
        if not is_large_node:
            return 0
        return (
            self.prev_mis_rates.get(node, 1.0)
            * self.node_to_count.get(node, node.ref_counter[gpu])
            * self.prefill_time(node.num_tokens, node.context_length)
        )

    def migrate_node_cost(self, node: TreeNode, from_gpu, to_gpu):
        cost = self.per_node_prefill_cost[node]

        self.current_prefill_cost_per_gpu[from_gpu] -= cost
        self.current_prefill_cost_per_gpu[to_gpu] += cost

        total_decoding_length = self.per_node_total_decode_lengths[node]
        self.current_decode_lengths_per_gpu[from_gpu] -= total_decoding_length
        self.current_decode_lengths_per_gpu[to_gpu] += total_decoding_length


class TTFTWindowedOverloadedDetector:

    def __init__(self, window_duration=timedelta(minutes=3)):
        self.data = {}
        self.window_duration = window_duration
        self.half_window_duration = window_duration / 2

    def add_data_point(self, timestamp, node, gpu, value):
        """Add a new data point and remove outdated entries."""
        key = (node, gpu)
        if key not in self.data:
            self.data[key] = deque()
        self.data[key].append((timestamp, value))
        self.purge_old_data(key, timestamp)

    def purge_old_data(self, key, current_time):
        """Remove data points that are older than the time window."""
        while (
            self.data[key]
            and self.data[key][0][0] < current_time - self.window_duration
        ):
            self.data[key].popleft()

    def rename_node(self, old_node, new_node, runtime_idx):
        old_key = (old_node, runtime_idx)
        new_key = (new_node, runtime_idx)
        if old_key in self.data:
            self.data[new_key] = self.data.pop(old_key)

    def calculate_half_window_averages(self, key):
        """Calculate averages for the first and second halves of the window."""
        first_half = []
        second_half = []
        half_window_cutoff = datetime.now() - self.half_window_duration
        if key not in self.data:
            return None, None
        for timestamp, value in self.data[key]:
            if timestamp < half_window_cutoff:
                first_half.append(value)
            else:
                second_half.append(value)
        if len(first_half) == 0:
            return None, None
        if len(second_half) == 0:
            return None, None
        avg_first_half = sum(first_half) / len(first_half) if first_half else 0
        avg_second_half = sum(second_half) / len(second_half) if second_half else 0

        return avg_first_half, avg_second_half

    def delete_after_allocation(self, node, gpu):
        key = (node, gpu)
        if key in self.data:
            del self.data[key]

    def is_node_overloaded(self, node, gpu):
        """Check if node is overloaded"""
        key = (node, gpu)
        avg_first_half, avg_second_half = self.calculate_half_window_averages(key)
        if avg_first_half is None and avg_second_half is None:
            return False
        return avg_second_half >= 2 * avg_first_half
