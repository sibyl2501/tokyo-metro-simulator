"""Dijkstra & Yen's K-shortest route search engine for Tokyo Metro.

Supports:
- Lexicographic priority modes:
  - 'fewest_transfers': minimize (transfers, stops)
  - 'fewest_stops': minimize (stops, transfers)
- Disabling specific transfers while keeping pass-through trains operational
- Closing specific track segments (both directions)
- Diverse candidate pool generation & before/after comparison
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.metro_graph import Edge, MetroGraph


@dataclass
class RouteLeg:
    line_id: str
    line_name: str
    direction_label: str
    node_ids: List[str]
    station_names: List[str]
    stops: int
    is_branch_leg: bool = False


@dataclass
class TransferStep:
    transfer_id: str
    transfer_label: str
    from_node_id: str
    to_node_id: str
    from_station: str
    to_station: str
    from_line_id: str
    to_line_id: str
    is_diff_name: bool
    stop_index_before: int


@dataclass
class RouteCandidate:
    origin_station: str
    destination_station: str
    node_ids: List[str]
    edges: List[Edge]
    stops: int
    transfers: int
    diff_name_transfers: int
    transfer_timing: float
    legs: List[RouteLeg] = field(default_factory=list)
    transfer_steps: List[TransferStep] = field(default_factory=list)
    title: str = ""
    guidance: str = ""
    score: Optional[float] = None

    @property
    def signature(self) -> str:
        return "->".join(self.node_ids)

    @property
    def transfer_ids(self) -> List[str]:
        return [step.transfer_id for step in self.transfer_steps]

    @property
    def segment_ids(self) -> List[str]:
        return [e.segment_id for e in self.edges if e.kind == "ride" and e.segment_id]

    @property
    def feature_vector(self) -> List[float]:
        """Return 4 interpretable numerical features for PyTorch preference learning.

        Features:
          0. stops: number of ride segments
          1. transfers: number of line transfers
          2. diff_name_transfers: number of transfers between differently named stations
          3. transfer_timing: (first_leg_stops / stops - 0.5) if transfers > 0 else 0.0
        """
        return [
            float(self.stops),
            float(self.transfers),
            float(self.diff_name_transfers),
            float(self.transfer_timing),
        ]

    def to_log_dict(self) -> dict:
        return {
            "signature": self.signature,
            "title": self.title,
            "guidance": self.guidance,
            "stops": self.stops,
            "transfers": self.transfers,
            "diff_name_transfers": self.diff_name_transfers,
            "transfer_timing": round(self.transfer_timing, 4),
            "transfer_ids": self.transfer_ids,
            "features": self.feature_vector,
        }


def _cost_tuple(stops: int, transfers: int, priority: str) -> Tuple[int, int]:
    if priority == "fewest_stops":
        return (stops, transfers)
    return (transfers, stops)


def _edge_allowed(
    edge: Edge,
    disabled_transfers: Set[str],
    closed_segments: Set[str],
    blocked_edges: Set[Tuple[str, str]],
) -> bool:
    if (edge.source, edge.target) in blocked_edges:
        return False
    if edge.kind == "transfer" and edge.transfer_id in disabled_transfers:
        return False
    if edge.kind == "ride" and edge.segment_id in closed_segments:
        return False
    return True


def dijkstra_shortest_path(
    graph: MetroGraph,
    start_nodes: Sequence[str],
    goal_nodes: Set[str],
    priority: str = "fewest_transfers",
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
    blocked_nodes: Optional[Set[str]] = None,
    blocked_edges: Optional[Set[Tuple[str, str]]] = None,
    blocked_station_names: Optional[Set[str]] = None,
    forbid_initial_transfer: bool = False,
) -> Optional[Tuple[List[str], List[Edge], int, int]]:
    """Run Dijkstra over the station-line graph with simple-path constraints.

    State includes `last_was_transfer` to disallow consecutive transfer edges
    (which would be redundant bounces without riding a train).
    """
    disabled_tr = set(disabled_transfers or ())
    closed_seg = set(closed_segments or ())
    b_nodes = set(blocked_nodes or ())
    b_edges = set(blocked_edges or ())
    b_stations = set(blocked_station_names or ())

    # Priority queue entries: (cost_a, cost_b, tie_idx, node_id, last_was_transfer, stops, transfers)
    pq: List[Tuple[int, int, int, str, bool, int, int]] = []
    best: Dict[Tuple[str, bool], Tuple[int, int]] = {}
    parent: Dict[Tuple[str, bool], Tuple[Optional[Tuple[str, bool]], Optional[Edge]]] = {}

    counter = 0
    for s_node in start_nodes:
        if s_node in b_nodes:
            continue
        state = (s_node, forbid_initial_transfer)
        init_cost = (0, 0)
        best[state] = init_cost
        parent[state] = (None, None)
        heapq.heappush(pq, (init_cost[0], init_cost[1], counter, s_node, forbid_initial_transfer, 0, 0))
        counter += 1

    while pq:
        c0, c1, _, curr_node, last_was_transfer, stops, transfers = heapq.heappop(pq)
        curr_state = (curr_node, last_was_transfer)
        if best.get(curr_state) != (c0, c1):
            continue

        if curr_node in goal_nodes:
            # Reconstruct path
            path_nodes: List[str] = []
            path_edges: List[Edge] = []
            walk: Optional[Tuple[str, bool]] = curr_state
            while walk is not None:
                path_nodes.append(walk[0])
                prev_state, edge = parent[walk]
                if edge is not None:
                    path_edges.append(edge)
                walk = prev_state
            path_nodes.reverse()
            path_edges.reverse()
            return path_nodes, path_edges, stops, transfers

        for edge in graph.adjacency.get(curr_node, []):
            nxt = edge.target
            if nxt in b_nodes:
                continue
            if not _edge_allowed(edge, disabled_tr, closed_seg, b_edges):
                continue

            # Disallow consecutive transfer edges
            if edge.kind == "transfer" and last_was_transfer:
                continue

            # Respect blocked station names (used by Yen's spur search to prevent station loops)
            nxt_station = graph.nodes[nxt].station_name
            curr_station = graph.nodes[curr_node].station_name
            if nxt_station in b_stations and nxt_station != curr_station and nxt not in goal_nodes:
                continue

            new_stops = stops + (1 if edge.kind == "ride" else 0)
            new_transfers = transfers + (1 if edge.kind == "transfer" else 0)
            new_cost = _cost_tuple(new_stops, new_transfers, priority)
            nxt_state = (nxt, edge.kind == "transfer")

            prev_best = best.get(nxt_state)
            if prev_best is None or new_cost < prev_best:
                best[nxt_state] = new_cost
                parent[nxt_state] = (curr_state, edge)
                heapq.heappush(
                    pq,
                    (new_cost[0], new_cost[1], counter, nxt, edge.kind == "transfer", new_stops, new_transfers),
                )
                counter += 1

    return None


def _is_realistic_simple_path(graph: MetroGraph, node_ids: List[str], edges: List[Edge]) -> bool:
    """Filter out unrealistic loops (revisiting an earlier station after leaving it)."""
    if len(set(node_ids)) != len(node_ids):
        return False

    # Never start or end with a transfer to the same station name
    if edges and edges[0].kind == "transfer":
        if graph.nodes[node_ids[0]].station_name == graph.nodes[node_ids[1]].station_name:
            return False
    if edges and edges[-1].kind == "transfer":
        if graph.nodes[node_ids[-2]].station_name == graph.nodes[node_ids[-1]].station_name:
            return False

    visited_stations: Set[str] = set()
    prev_station: Optional[str] = None
    for idx, nid in enumerate(node_ids):
        st_name = graph.nodes[nid].station_name
        if st_name != prev_station:
            if st_name in visited_stations:
                return False
            visited_stations.add(st_name)
        if idx > 0 and edges[idx - 1].kind == "transfer":
            # Ensure we don't have two consecutive transfers
            if idx >= 2 and edges[idx - 2].kind == "transfer":
                return False
        prev_station = st_name
    return True


def yen_k_shortest_paths(
    graph: MetroGraph,
    origin_station: str,
    destination_station: str,
    priority: str = "fewest_transfers",
    k: int = 5,
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
) -> List[Tuple[List[str], List[Edge], int, int]]:
    """Yen's K-shortest loopless paths on top of Dijkstra."""
    if origin_station == destination_station:
        start_node = graph.station_name_to_nodes[origin_station][0]
        return [([start_node], [], 0, 0)]

    start_nodes = graph.station_name_to_nodes[origin_station]
    goal_nodes = set(graph.station_name_to_nodes[destination_station])
    disabled_tr = set(disabled_transfers or ())
    closed_seg = set(closed_segments or ())

    first = dijkstra_shortest_path(
        graph=graph,
        start_nodes=start_nodes,
        goal_nodes=goal_nodes,
        priority=priority,
        disabled_transfers=disabled_tr,
        closed_segments=closed_seg,
    )
    if first is None:
        return []

    accepted: List[Tuple[List[str], List[Edge], int, int]] = [first]
    accepted_sigs: Set[str] = {"->".join(first[0])}
    # Heap of candidate paths: (cost0, cost1, tie_idx, nodes, edges, stops, transfers)
    potential: List[Tuple[int, int, int, List[str], List[Edge], int, int]] = []
    potential_sigs: Set[str] = set()
    tie = 0

    for _ in range(1, k * 3):
        if len(accepted) >= k:
            break
        prev_nodes, prev_edges, _, _ = accepted[-1]

        for spur_idx in range(len(prev_nodes) - 1):
            spur_node = prev_nodes[spur_idx]
            root_nodes = prev_nodes[: spur_idx + 1]
            root_edges = prev_edges[:spur_idx]

            blocked_edges: Set[Tuple[str, str]] = set()
            for acc_nodes, _, _, _ in accepted:
                if len(acc_nodes) > spur_idx + 1 and acc_nodes[: spur_idx + 1] == root_nodes:
                    blocked_edges.add((acc_nodes[spur_idx], acc_nodes[spur_idx + 1]))

            blocked_nodes = set(root_nodes[:-1])
            blocked_stations = {graph.nodes[nid].station_name for nid in root_nodes[:-1]}
            forbid_initial_transfer = bool(root_edges and root_edges[-1].kind == "transfer")

            spur_res = dijkstra_shortest_path(
                graph=graph,
                start_nodes=[spur_node],
                goal_nodes=goal_nodes,
                priority=priority,
                disabled_transfers=disabled_tr,
                closed_segments=closed_seg,
                blocked_nodes=blocked_nodes,
                blocked_edges=blocked_edges,
                blocked_station_names=blocked_stations,
                forbid_initial_transfer=forbid_initial_transfer,
            )
            if spur_res is None:
                continue

            spur_nodes, spur_edges, _, _ = spur_res
            total_nodes = root_nodes[:-1] + spur_nodes
            total_edges = root_edges + spur_edges

            if not _is_realistic_simple_path(graph, total_nodes, total_edges):
                continue

            sig = "->".join(total_nodes)
            if sig in accepted_sigs or sig in potential_sigs:
                continue

            total_stops = sum(1 for e in total_edges if e.kind == "ride")
            total_transfers = sum(1 for e in total_edges if e.kind == "transfer")
            c0, c1 = _cost_tuple(total_stops, total_transfers, priority)
            heapq.heappush(
                potential,
                (c0, c1, tie, total_nodes, total_edges, total_stops, total_transfers),
            )
            potential_sigs.add(sig)
            tie += 1

        if not potential:
            break

        _, _, _, best_nodes, best_edges, best_stops, best_transfers = heapq.heappop(potential)
        best_sig = "->".join(best_nodes)
        if best_sig not in accepted_sigs:
            accepted.append((best_nodes, best_edges, best_stops, best_transfers))
            accepted_sigs.add(best_sig)

    return accepted


def _build_route_candidate(
    graph: MetroGraph,
    origin_station: str,
    destination_station: str,
    node_ids: List[str],
    edges: List[Edge],
    stops: int,
    transfers: int,
) -> RouteCandidate:
    legs: List[RouteLeg] = []
    transfer_steps: List[TransferStep] = []

    if not edges:
        node = graph.nodes[node_ids[0]]
        line = graph.lines[node.line_id]
        legs.append(
            RouteLeg(
                line_id=line.id,
                line_name=line.name,
                direction_label="",
                node_ids=[node.id],
                station_names=[node.station_name],
                stops=0,
            )
        )
    else:
        curr_leg_nodes: List[str] = [node_ids[0]]
        curr_line_id: Optional[str] = None
        curr_dir: Optional[str] = None
        curr_is_branch = False
        stops_so_far = 0

        def flush_leg() -> None:
            nonlocal curr_leg_nodes, curr_line_id, curr_dir, curr_is_branch
            if len(curr_leg_nodes) >= 2 and curr_line_id is not None:
                line = graph.lines[curr_line_id]
                # Refine Marunouchi direction if riding from Honancho branch through to Ikebukuro
                refined_dir = curr_dir or ""
                if curr_line_id == "M" and refined_dir == "中野坂上・池袋方面":
                    end_node = curr_leg_nodes[-1]
                    if end_node.startswith("M-") and int(end_node.split("-")[1]) >= 7:
                        refined_dir = "池袋方面"
                    else:
                        refined_dir = "中野坂上方面"
                legs.append(
                    RouteLeg(
                        line_id=line.id,
                        line_name=line.name,
                        direction_label=refined_dir,
                        node_ids=list(curr_leg_nodes),
                        station_names=[graph.nodes[nid].station_name for nid in curr_leg_nodes],
                        stops=len(curr_leg_nodes) - 1,
                        is_branch_leg=curr_is_branch,
                    )
                )

        for idx, edge in enumerate(edges):
            if edge.kind == "ride":
                src_node = graph.nodes[edge.source]
                dst_node = graph.nodes[edge.target]

                # Detect Marunouchi branch reversal at Nakano-sakaue (e.g. Honancho <-> Ogikubo)
                # (Mb-05 -> M-06 -> M-05 or M-05 -> M-06 -> Mb-05)
                branch_turn = False
                if curr_line_id == "M" and len(curr_leg_nodes) >= 2:
                    prev_node_id = curr_leg_nodes[-2]
                    if edge.source == "M-06":
                        if (prev_node_id == "Mb-05" and edge.target == "M-05") or (
                            prev_node_id == "M-05" and edge.target == "Mb-05"
                        ):
                            branch_turn = True

                if branch_turn:
                    flush_leg()
                    curr_leg_nodes = [edge.source, edge.target]
                    curr_line_id = edge.line_id
                    curr_dir = edge.direction_label
                    curr_is_branch = src_node.branch == "honancho" or dst_node.branch == "honancho"
                else:
                    if curr_line_id is None:
                        curr_line_id = edge.line_id
                        curr_dir = edge.direction_label
                        curr_is_branch = src_node.branch == "honancho" or dst_node.branch == "honancho"
                    else:
                        # Update direction when entering Honancho branch from M-06 (Ikebukuro -> Honancho)
                        if edge.direction_label == "方南町方面":
                            curr_dir = "方南町方面"
                        elif curr_dir == "中野坂上・池袋方面" and edge.direction_label == "池袋方面":
                            curr_dir = "池袋方面"
                        if src_node.branch == "honancho" or dst_node.branch == "honancho":
                            curr_is_branch = True
                    curr_leg_nodes.append(edge.target)
                stops_so_far += 1
            else:
                # Transfer edge
                flush_leg()
                src_node = graph.nodes[edge.source]
                dst_node = graph.nodes[edge.target]
                transfer_steps.append(
                    TransferStep(
                        transfer_id=edge.transfer_id or "",
                        transfer_label=edge.transfer_label or src_node.station_name,
                        from_node_id=edge.source,
                        to_node_id=edge.target,
                        from_station=src_node.station_name,
                        to_station=dst_node.station_name,
                        from_line_id=src_node.line_id,
                        to_line_id=dst_node.line_id,
                        is_diff_name=edge.is_diff_name,
                        stop_index_before=stops_so_far,
                    )
                )
                curr_leg_nodes = [edge.target]
                curr_line_id = None
                curr_dir = None
                curr_is_branch = False

        flush_leg()

    diff_name_transfers = sum(1 for t in transfer_steps if t.is_diff_name)
    if transfers > 0 and stops > 0 and transfer_steps:
        avg_pos = sum(t.stop_index_before / stops for t in transfer_steps) / len(transfer_steps)
        transfer_timing = avg_pos - 0.5
    else:
        transfer_timing = 0.0

    from src.formatter import format_route_guidance, format_route_title

    candidate = RouteCandidate(
        origin_station=origin_station,
        destination_station=destination_station,
        node_ids=node_ids,
        edges=edges,
        stops=stops,
        transfers=transfers,
        diff_name_transfers=diff_name_transfers,
        transfer_timing=transfer_timing,
        legs=legs,
        transfer_steps=transfer_steps,
    )
    candidate.title = format_route_title(candidate)
    candidate.guidance = format_route_guidance(candidate)
    return candidate


def search_routes(
    graph: MetroGraph,
    origin_station: str,
    destination_station: str,
    priority: str = "fewest_transfers",
    max_candidates: int = 5,
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
) -> List[RouteCandidate]:
    """Generate a diverse, realistic set of candidate routes via Dijkstra + Yen's algorithm.

    Searches under both 'fewest_transfers' and 'fewest_stops' so Pareto trade-off
    candidates are always discovered, deduplicates them, and sorts according to
    the requested priority.
    """
    if origin_station not in graph.station_name_to_nodes:
        raise ValueError(f"Unknown origin station: {origin_station}")
    if destination_station not in graph.station_name_to_nodes:
        raise ValueError(f"Unknown destination station: {destination_station}")

    disabled_tr = set(disabled_transfers or ())
    closed_seg = set(closed_segments or ())

    raw_pool: Dict[str, Tuple[List[str], List[Edge], int, int]] = {}
    for mode in ("fewest_transfers", "fewest_stops"):
        paths = yen_k_shortest_paths(
            graph=graph,
            origin_station=origin_station,
            destination_station=destination_station,
            priority=mode,
            k=max(max_candidates, 5),
            disabled_transfers=disabled_tr,
            closed_segments=closed_seg,
        )
        for item in paths:
            sig = "->".join(item[0])
            if sig not in raw_pool:
                raw_pool[sig] = item

    if not raw_pool:
        return []

    candidates = [
        _build_route_candidate(graph, origin_station, destination_station, n, e, s, t)
        for (n, e, s, t) in raw_pool.values()
    ]

    # Filter out clearly excessive detour candidates compared to the shortest stop count
    min_stops = min(c.stops for c in candidates)
    min_transfers = min(c.transfers for c in candidates)
    filtered = [
        c
        for c in candidates
        if c.stops <= min_stops + 8 and c.transfers <= min_transfers + 2
    ]
    if not filtered:
        filtered = candidates

    if priority == "fewest_stops":
        filtered.sort(key=lambda c: (c.stops, c.transfers, c.diff_name_transfers, c.title))
    else:
        filtered.sort(key=lambda c: (c.transfers, c.stops, c.diff_name_transfers, c.title))

    return filtered[:max_candidates]
