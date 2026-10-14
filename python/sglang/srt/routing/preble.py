# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Token-level E2 routing, adapted from WukLab/preble (Apache-2.0).

The radix tree describes routing assignments, not authoritative GPU KV residency.
Reference: global_scheduler_with_time.py at 1a35eae20c416f7062ca81775aab72890323fd97.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Sequence


def a6000_prefill_time(tokens: int, context: int) -> float:
    """Original Mistral-7B/A6000 regression, in seconds; profile other hardware."""
    if tokens == 0:
        return 0.0
    if tokens >= 384:
        linear = 0.10842571 * tokens + 4.209777054806409
    elif tokens >= 192:
        linear = -118 + 1.25 * tokens - 2.56e-3 * tokens**2
    else:
        linear = 22
    attention = 0.32 if context <= 1024 else 1.86e-4 * context + 0.159
    quadratic = 0.0
    if tokens * context > 1024 * 1024:
        quadratic = 1.13e-3 * tokens + 1.75e-3 * context + 2.19e-6 * tokens * context
    return (linear + attention + quadratic) / 900


@dataclass(eq=False)
class PrefixNode:
    tokens: tuple[int, ...]
    parent: PrefixNode | None
    context: int
    children: dict[int, PrefixNode] = field(default_factory=dict)
    workers: set[int] = field(default_factory=set)
    active: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_used: float = 0.0

    @property
    def large(self) -> bool:
        return len(self.tokens) > self.context - len(self.tokens)

    def path(self):
        node = self
        while node.parent is not None:
            yield node
            node = node.parent


@dataclass
class Assignment:
    leaf: PrefixNode
    worker: int
    output_budget: int
    finished: bool = False


@dataclass
class Observation:
    timestamp: float
    leaf: PrefixNode
    prompt_tokens: int
    hit_tokens: int
    output_budget: int


class PrebleScheduler:
    """Single-model, fixed-worker E2 scheduler; callers serialize state access.

    Exploitation follows the existing prefix assignment when matched context is
    larger than the new suffix. Exploration minimizes windowed prefill cost plus
    outstanding decode cost. Node splitting preserves active references; window
    statistics are grouped against the current tree, including after a split.
    """

    def __init__(
        self,
        num_workers: int,
        window_seconds: float = 180,
        prefill_time: Callable[[int, int], float] = a6000_prefill_time,
        initial_tpot: float = 0.15,
        max_nodes: int = 100000,
        enable_rebalancing: bool = True,
    ):
        if num_workers < 1 or window_seconds <= 0 or initial_tpot <= 0 or max_nodes < 2:
            raise ValueError(
                "Worker count, window, TPOT and tree limit must be positive"
            )
        self.num_workers = num_workers
        self.window_seconds = window_seconds
        self.prefill_time = prefill_time
        self.max_nodes = max_nodes
        self.root = PrefixNode((), None, 0)
        self.nodes = {self.root}
        self.history = deque()
        self.tpot = [deque([initial_tpot], maxlen=200) for _ in range(num_workers)]
        self.scheduled = 0
        self.enable_rebalancing = enable_rebalancing
        self.ttft_history = deque()
        self.migrations = 0
        self.replications = 0
        self.outstanding_tokens = [0] * num_workers

    def _insert(self, tokens: Sequence[int]) -> PrefixNode:
        node = self.root
        offset = 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                child = PrefixNode(tuple(tokens[offset:]), node, len(tokens))
                node.children[tokens[offset]] = child
                self.nodes.add(child)
                return child
            matched = 0
            for left, right in zip(child.tokens, tokens[offset:]):
                if left != right:
                    break
                matched += 1
            if matched < len(child.tokens):
                split = PrefixNode(child.tokens[:matched], node, node.context + matched)
                split.workers = child.workers.copy()
                split.active.update(child.active)
                split.last_used = child.last_used
                node.children[split.tokens[0]] = split
                child.tokens = child.tokens[matched:]
                child.parent = split
                split.children[child.tokens[0]] = child
                self.nodes.add(split)
                node = split
            else:
                node = child
            offset += matched
        return node

    @staticmethod
    def important(node: PrefixNode) -> PrefixNode:
        while node.parent is not None and not node.large:
            node = node.parent
        return node

    def _statistics(self, now: float):
        while self.history and self.history[0].timestamp < now - self.window_seconds:
            self.history.popleft()
        while self.ttft_history and self.ttft_history[0][0] < now - self.window_seconds:
            self.ttft_history.popleft()
        for node in self.nodes:
            if node.last_used < now - self.window_seconds and not any(
                node.active.values()
            ):
                node.workers.clear()
        stats = {}
        for event in self.history:
            node = self.important(event.leaf)
            entry = stats.setdefault(node, [0, 0, 0, event.output_budget])
            entry[0] += 1
            entry[1] += event.prompt_tokens
            entry[2] += event.hit_tokens
            entry[3] = event.output_budget
        return stats

    def _costs(self, stats, minimum_count: int = 1):
        # Decode load survives history expiry and prefix ownership migration.
        costs = [
            tokens * median(self.tpot[worker])
            for worker, tokens in enumerate(self.outstanding_tokens)
        ]
        node_costs = {}
        for node, (count, total, hits, output) in stats.items():
            if not node.workers or count < minimum_count:
                continue
            prefill = (
                (1 - hits / total)
                * count
                * self.prefill_time(len(node.tokens), node.context)
                / len(node.workers)
            )
            node_costs[node] = {}
            active_workers = {worker for worker, count in node.active.items() if count}
            for worker in node.workers | active_workers:
                cost = (prefill if worker in node.workers else 0) + node.active[
                    worker
                ] * output * median(self.tpot[worker])
                if worker in node.workers:
                    costs[worker] += prefill
                if worker in node.workers:
                    node_costs[node][worker] = cost
        return costs, node_costs

    def _prune(self):
        # Keep both in-flight and windowed paths; remove old leaves under pressure.
        if len(self.nodes) <= self.max_nodes:
            return
        protected = {self.root}
        for event in self.history:
            protected.update(event.leaf.path())
        leaves = deque(n for n in self.nodes if not n.children and n not in protected)
        while leaves and len(self.nodes) > self.max_nodes:
            node = leaves.popleft()
            if any(node.active.values()):
                continue
            parent = node.parent
            del parent.children[node.tokens[0]]
            self.nodes.remove(node)
            if parent not in protected and not parent.children:
                leaves.append(parent)

    def select(
        self,
        tokens: Sequence[int],
        output_budget: int,
        available: Sequence[int] | None = None,
    ) -> Assignment:
        if not tokens or output_budget < 0:
            raise ValueError(
                "Nonempty input tokens and nonnegative output budget required"
            )
        candidates = list(range(self.num_workers) if available is None else available)
        if not candidates or any(w < 0 or w >= self.num_workers for w in candidates):
            raise ValueError("No valid available worker")
        now = time.monotonic()
        leaf = self._insert(tokens)
        # A split changes important nodes, so regroup historical observations.
        stats = self._statistics(now)
        costs, _ = self._costs(stats)
        path = list(leaf.path())
        matched = max(
            (node.context for node in path if node.workers.intersection(candidates)),
            default=0,
        )
        if len(tokens) - matched < matched:
            node = leaf
            while node.parent is not None:
                assigned = node.workers.intersection(candidates)
                if assigned:
                    candidates = sorted(assigned)
                    break
                node = node.parent
        worker = min(candidates, key=lambda w: (costs[w], w))
        self.outstanding_tokens[worker] += output_budget
        hit_tokens = max(
            (node.context for node in path if worker in node.workers), default=0
        )
        for node in path:
            node.workers.add(worker)
            node.active[worker] += 1
            node.last_used = now
        self.history.append(
            Observation(now, leaf, len(tokens), hit_tokens, output_budget)
        )
        self.scheduled += 1
        if self.enable_rebalancing and self.scheduled >= 50 * self.num_workers:
            if len(list(leaf.path())) - len(list(self.important(leaf).path())) < 3:
                self._rebalance(now)
        self._prune()
        return Assignment(leaf, worker, output_budget)

    def finish(
        self,
        assignment: Assignment,
        output_tokens: int = 0,
        ttft: float = 0,
        latency: float = 0,
    ):
        """Release once, including errors and cancellations; update measured TPOT."""
        if assignment.finished:
            return
        assignment.finished = True
        self.outstanding_tokens[assignment.worker] -= assignment.output_budget
        for node in assignment.leaf.path():
            node.active[assignment.worker] -= 1
        if output_tokens > 1 and latency > ttft > 0:
            self.tpot[assignment.worker].append((latency - ttft) / (output_tokens - 1))
        if ttft > 0:
            self.ttft_history.append(
                (time.monotonic(), assignment.leaf, assignment.worker, ttft)
            )

    def _overloaded(self, node, worker, now):
        while self.ttft_history and self.ttft_history[0][0] < now - self.window_seconds:
            self.ttft_history.popleft()
        halves = ([], [])
        for timestamp, leaf, selected, ttft in self.ttft_history:
            if selected == worker and self.important(leaf) is node:
                halves[int(timestamp >= now - self.window_seconds / 2)].append(ttft)
        return bool(halves[0] and halves[1]) and (
            sum(halves[1]) / len(halves[1]) >= 2 * sum(halves[0]) / len(halves[0])
        )

    @staticmethod
    def _assign_subtree(node, worker, replicate=False):
        pending = [node]
        while pending:
            current = pending.pop()
            if replicate:
                current.workers.add(worker)
            else:
                current.workers = {worker}
            pending.extend(current.children.values())

    def _rebalance(self, now):
        """Move small prefix groups; replicate an indivisible, overloaded group.

        Like upstream, this only changes future routing. Active requests remain
        on their original workers, and no KV bytes are transferred.
        """
        stats = self._statistics(now)
        loads, node_costs = self._costs(stats, minimum_count=2)
        ranked = sorted(range(self.num_workers), key=lambda w: (-loads[w], w))
        while len(ranked) > 1:
            high, low = ranked[0], ranked[-1]
            if loads[high] <= 1.5 * loads[low]:
                break
            groups = sorted(
                (
                    (costs[high], node)
                    for node, costs in node_costs.items()
                    if high in costs
                ),
                key=lambda item: item[0],
            )
            if len(groups) == 1:
                cost, node = groups[0]
                if low not in node.workers and self._overloaded(node, high, now):
                    self._assign_subtree(node, low, replicate=True)
                    self.replications += 1
                    self.ttft_history = deque(
                        e
                        for e in self.ttft_history
                        if not (e[2] == high and self.important(e[1]) is node)
                    )
            else:
                for cost, node in groups:
                    if loads[high] - cost < loads[low] + cost:
                        break
                    self._assign_subtree(node, low)
                    self.migrations += 1
                    loads[high] -= cost
                    loads[low] += cost
                    if loads[high] <= 1.5 * loads[low]:
                        break
            ranked = ranked[1:]
