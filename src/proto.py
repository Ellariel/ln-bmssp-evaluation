from __future__ import annotations
from dataclasses import dataclass
from bisect import bisect_left
from typing import List, Tuple, Dict, Optional
import heapq
import networkx as nx



# ---------------------------------------------------------------------------
# Block and BlockList implementation
# ---------------------------------------------------------------------------

NodeId = int
Cost = float


@dataclass
class PullResult:
    nodes: List[NodeId]
    min_upper_bound: Cost


class Block:
    __slots__ = ("capacity", "items", "upper_bound")

    def __init__(self, capacity: int, items: Optional[List[Tuple[NodeId, Cost]]] = None):
        self.capacity = capacity
        self.items: List[Tuple[NodeId, Cost]] = sorted(items or [], key=lambda x: x[1])
        self.upper_bound: Cost = (self.items[-1][1] if self.items else float("-inf"))

    def __len__(self) -> int:
        return len(self.items)

    def _recompute_upper_bound(self) -> None:
        self.upper_bound = (self.items[-1][1] if self.items else float("-inf"))

    def add(self, node_id: NodeId, cost: Cost) -> Tuple[Optional["Block"], Optional["Block"]]:
        idx = bisect_left([c for _, c in self.items], cost)
        self.items.insert(idx, (node_id, cost))
        if len(self.items) <= self.capacity:
            self._recompute_upper_bound()
            return (self, None)
        mid = len(self.items) // 2
        left = Block(self.capacity, self.items[:mid])
        right = Block(self.capacity, self.items[mid:])
        return (left, right)

    def extend_front(self, pairs: List[Tuple[NodeId, Cost]]) -> List["Block"]:
        if not pairs:
            return [self]
        all_items = pairs + self.items
        all_items.sort(key=lambda x: x[1])
        blocks: List[Block] = []
        for i in range(0, len(all_items), self.capacity):
            blocks.append(Block(self.capacity, all_items[i:i + self.capacity]))
        return blocks

    def pop_min_many(self, k: int) -> List[Tuple[NodeId, Cost]]:
        if not self.items or k <= 0:
            return []
        take = min(k, len(self.items))
        out = self.items[:take]
        self.items = self.items[take:]
        self._recompute_upper_bound()
        return out


class BlockList:
    def __init__(self, M: int, initial_upper_bound: Cost = float("inf")) -> None:
        if M <= 0:
            raise ValueError("Block capacity M must be positive")
        self.M = M
        self.blocks: List[Block] = []
        self.global_upper_bound: Cost = initial_upper_bound

    def _block_upper_bounds(self) -> List[Cost]:
        return [b.upper_bound for b in self.blocks]

    def _find_block_index_for_cost(self, cost: Cost) -> int:
        ubs = self._block_upper_bounds()
        return bisect_left(ubs, cost)

    def insert(self, node_id: NodeId, cost: Cost) -> None:
        idx = self._find_block_index_for_cost(cost)
        if idx == len(self.blocks):
            if self.blocks and len(self.blocks[-1]) < self.M and cost >= self.blocks[-1].items[-1][1]:
                blk = self.blocks[-1]
                left, right = blk.add(node_id, cost)
                if right is not None:
                    self.blocks.pop()
                    self.blocks.append(left)
                    self.blocks.append(right)
            else:
                self.blocks.append(Block(self.M, [(node_id, cost)]))
            return
        blk = self.blocks[idx]
        left, right = blk.add(node_id, cost)
        if right is None:
            self.blocks[idx] = left
        else:
            self.blocks[idx] = left
            self.blocks.insert(idx + 1, right)

    def batch_prepend(self, items: List[Tuple[NodeId, Cost]]) -> None:
        if not items:
            return
        items = sorted(items, key=lambda x: x[1])
        if not self.blocks:
            for i in range(0, len(items), self.M):
                self.blocks.append(Block(self.M, items[i:i + self.M]))
            return
        first = self.blocks.pop(0)
        new_blocks = first.extend_front(items)
        self.blocks = new_blocks + self.blocks

    def pull(self, limit: int) -> PullResult:
        if limit <= 0 or not self.blocks:
            return PullResult(nodes=[], min_upper_bound=(self.blocks[0].upper_bound if self.blocks else float("inf")))
        taken_nodes: List[NodeId] = []
        remaining = limit
        i = 0
        while i < len(self.blocks) and remaining > 0:
            blk = self.blocks[i]
            popped = blk.pop_min_many(remaining)
            if popped:
                taken_nodes.extend(n for n, _ in popped)
                remaining -= len(popped)
            if len(blk) == 0:
                self.blocks.pop(i)
            else:
                i += 1
        min_ub = self.blocks[0].upper_bound if self.blocks else float("inf")
        return PullResult(nodes=taken_nodes, min_upper_bound=min_ub)

    def __len__(self) -> int:
        return sum(len(b) for b in self.blocks)

    def is_empty(self) -> bool:
        return not self.blocks

    def peek_min_upper_bound(self) -> Cost:
        return self.blocks[0].upper_bound if self.blocks else float("inf")

    def debug_dump(self) -> List[List[Tuple[NodeId, Cost]]]:
        return [b.items[:] for b in self.blocks]


# ---------------------------------------------------------------------------
# BMSSP implementation (distance-only) using BlockList
# ---------------------------------------------------------------------------

def bmssp_distance_only(graph: Dict[NodeId, List[Tuple[NodeId, Cost]]],
                        source: NodeId,
                        block_capacity: int = 4,
                        batch_size: int = 16) -> Dict[NodeId, Cost]:
    dist: Dict[NodeId, Cost] = {source: 0.0}
    visited: set[NodeId] = set()
    frontier = BlockList(block_capacity)
    frontier.insert(source, 0.0)

    while not frontier.is_empty():
        pulled = frontier.pull(batch_size)
        layer_nodes = pulled.nodes
        for u in layer_nodes:
            if u in visited:
                continue
            visited.add(u)
            du = dist[u]
            for v, w in graph.get(u, []):
                nd = du + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    frontier.insert(v, nd)
    return dist


# ---------------------------------------------------------------------------
# Baseline Dijkstra
# ---------------------------------------------------------------------------

def dijkstra_distance_only(graph: Dict[NodeId, List[Tuple[NodeId, Cost]]],
                           source: NodeId) -> Dict[NodeId, Cost]:
    dist: Dict[NodeId, Cost] = {source: 0.0}
    pq: List[Tuple[Cost, NodeId]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ---------------------------------------------------------------------------
# NetworkX Dijkstra
# ---------------------------------------------------------------------------

def dijkstrax_distance_only(graph: nx.Graph,
                            source: NodeId) -> Dict[NodeId, Cost]:

    return nx.single_source_dijkstra_path_length(graph, source)


