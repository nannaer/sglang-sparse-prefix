# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Virtual cache adapted from WukLab/preble (Apache-2.0).

Source: preble/global_lru_cache.py at
1a35eae20c416f7062ca81775aab72890323fd97. Node ordering, timestamp updates,
splits and virtual eviction preserve the reference implementation's behavior.
"""

import copy
import heapq
import time
from collections import defaultdict, deque
from typing import Optional
from uuid import uuid4


class TreeNode:
    def __init__(self, num_nodes):
        self.id = uuid4()
        self.children = defaultdict(TreeNode)
        self.parent: Optional[TreeNode] = None
        self.value = None
        self.key = None
        self.ref_counter = [0 for _ in range(num_nodes)]
        self.load = 0
        self.last_access_time = time.time()
        self.evicted_gpus = set()
        self.cached_gpus = set()
        self.is_leaf = False
        self.decode_length = deque()
        self.context_length = 0
        self.depth = 0

    def has_cached_gpu(self, gpu):
        return gpu in self.cached_gpus and gpu not in self.evicted_gpus

    @property
    def context_so_far(self):
        return self.context_length - self.num_tokens

    @property
    def num_tokens(self):
        return len(self.value)

    def __lt__(self, other):
        if self.ref_counter == other.ref_counter:
            return self.last_access_time < other.last_access_time
        return self.ref_counter < other.ref_counter

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return (
            f"TreeNode(id={self.id}, ref_counter={self.ref_counter}, "
            f"cached_gpus={self.cached_gpus}, evicted_gpus:{self.evicted_gpus})"
        )


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class LPRadixCache:
    def __init__(self, histogram, disable=False, num_gpus=2, lock=None):
        self.num_gpus = num_gpus
        self.reset()
        self.disable = disable
        self.histogram = histogram
        self.num_iters = 0
        self.lock = lock
        self.updates = {}

    def reset(self):
        self.root_node = TreeNode(num_nodes=self.num_gpus)
        self.root_node.value = []
        self.root_node.key = []
        self.root_node.ref_counter = [1 for _ in range(self.num_gpus)]
        self.allocated_size_ = [0 for _ in range(self.num_gpus)]
        self.all_nodes = {self.root_node}

    def find_node(self, key):
        if self.disable:
            return None
        return self._match_prefix_helper(self.root_node, key)

    def _match_prefix_helper(self, node, key):
        node.last_access_time = time.time()
        if len(key) == 0:
            return node
        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = match(child.key, key)
            if prefix_len < len(child.key):
                assert False
                return {}, None
            return self._match_prefix_helper(child, key[prefix_len:])
        return node

    def insert(self, key, value=None, all_modified_nodes=None, split_nodes=None):
        if all_modified_nodes is None:
            all_modified_nodes = set()
        if split_nodes is None:
            split_nodes = {}
        if self.disable:
            return len(key)
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, split_nodes=split_nodes)

    def _split_node(self, key, child: TreeNode, split_len):
        new_node = TreeNode(num_nodes=self.num_gpus)
        new_node.children = {key[split_len:][0]: child}
        new_node.key = child.key[:split_len]
        new_node.parent = child.parent
        new_node.load = child.load
        new_node.depth = child.depth
        new_node.cached_gpus = copy.deepcopy(child.cached_gpus)
        new_node.evicted_gpus = copy.deepcopy(child.evicted_gpus)
        new_node.ref_counter = copy.deepcopy(child.ref_counter)
        new_node.context_length = child.parent.context_length + split_len
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.depth = new_node.depth + 1
        new_node.parent.children[key[:split_len][0]] = new_node
        self.all_nodes.add(new_node)
        return new_node

    def _insert_helper(
        self, node: TreeNode, key, value, split_nodes, parent_context_length=0, depth=0
    ):
        node.last_access_time = time.time()
        node.load += 1
        if key and key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = match(child.key, key)
            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    child.load += 1
                    return child
                return self._insert_helper(
                    child,
                    key[prefix_len:],
                    value[prefix_len:],
                    split_nodes,
                    parent_context_length + prefix_len,
                    depth=depth + 1,
                )
            new_node = self._split_node(child.key, child, prefix_len)
            split_nodes[child] = new_node
            return self._insert_helper(
                new_node,
                key[prefix_len:],
                value[prefix_len:],
                split_nodes,
                parent_context_length + prefix_len,
                depth=depth + 2,
            )
        if len(key):
            new_node = TreeNode(num_nodes=self.num_gpus)
            new_node.cached_gpus = set()
            new_node.evicted_gpus = set()
            new_node.parent = node
            new_node.value = value
            new_node.key = copy.deepcopy(key)
            new_node.load = 1
            new_node.depth = depth + 1
            new_node.context_length = parent_context_length + len(key)
            node.children[key[0]] = new_node
            self.all_nodes.add(new_node)
            return new_node
        return node

    def update_allocated_size(self, node: TreeNode, runtime_id):
        while node:
            node.ref_counter[runtime_id] += 1
            if not node.has_cached_gpu(runtime_id):
                self.allocated_size_[runtime_id] += len(node.value)
                if runtime_id in node.evicted_gpus:
                    node.evicted_gpus.remove(runtime_id)
                node.cached_gpus.add(runtime_id)
            node = node.parent

    def dec_ref_counter(self, node, runtime_id):
        delta = 0
        while node != self.root_node:
            node.ref_counter[runtime_id] -= 1
            assert node.ref_counter[runtime_id] >= 0
            node = node.parent
        return delta

    def remove_completed_input_ids(self, input_ids, runtime_id):
        node = self.find_node(input_ids)
        self.dec_ref_counter(node, runtime_id)

    def allocated_size(self, runtime_id):
        return self.allocated_size_[runtime_id]

    def virtual_lru_eviction(self, num_new_tokens, runtime_id):
        leaves = self.collect_nodes_on_runtime_by_ref_cnt_and_access_time(runtime_id)
        heapq.heapify(leaves)
        num_evicted = 0
        visited = set()
        evicited = set()
        while num_evicted < num_new_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            num_evicted += len(x.value)
            evicited.add(x)
            visited.add(x)
            all_siblings_visited = True
            for child in x.parent.children.values():
                if (
                    child != x
                    and child.has_cached_gpu(runtime_id)
                    and child not in visited
                ):
                    all_siblings_visited = False
                    break
            if all_siblings_visited:
                heapq.heappush(leaves, x.parent)
        return evicited

    def collect_nodes_on_runtime_by_ref_cnt_and_access_time(self, runtime_id):
        priority_queue = []
        for node in self.all_nodes:
            if node.has_cached_gpu(runtime_id):
                heapq.heappush(priority_queue, node)
        return priority_queue

    def _collect_nodes(self):
        ret_list = []

        def dfs_(cur_node):
            ret_list.append(cur_node)
            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    def collected_nodes_with_runtime_idx(self, runtime_id):
        priority_queue = []
        current_allocated_size = 0
        for node in self._collect_nodes():
            if node.has_cached_gpu(runtime_id):
                current_allocated_size += len(node.value)
            if node.ref_counter[runtime_id] == 0 and node.has_cached_gpu(runtime_id):
                heapq.heappush(priority_queue, node)
        self.allocated_size_[runtime_id] = current_allocated_size
        return priority_queue

    def evict_with_runtime_id_without_removing(
        self, num_tokens, evict_callback, runtime_id
    ):
        if self.disable:
            raise RuntimeError()
        leaves = self.collected_nodes_with_runtime_idx(runtime_id)
        heapq.heapify(leaves)
        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            if x == self.root_node:
                break
            if x.ref_counter[runtime_id] > 0:
                continue
            if x.has_cached_gpu(runtime_id):
                self.allocated_size_[runtime_id] -= len(x.value)
                num_evicted += evict_callback(x)
                evicted_all_sibling_on_this_runtime = not any(
                    child.has_cached_gpu(runtime_id)
                    for child in x.parent.children.values()
                    if child != x
                )
                if evicted_all_sibling_on_this_runtime:
                    heapq.heappush(leaves, x.parent)

    def get_evictable_size(self, runtime_id):
        current_allocated_size = 0
        for node in self._collect_nodes():
            if node.ref_counter[runtime_id] == 0 and node.has_cached_gpu(runtime_id):
                current_allocated_size += len(node.value)
        return current_allocated_size
