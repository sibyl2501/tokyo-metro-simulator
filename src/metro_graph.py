"""Graph representation of the Tokyo Metro network.

Each station-line pair (e.g., '銀座・銀座線' = 'G-09', '銀座・丸ノ内線' = 'M-16')
is modeled as an independent node. Adjacent stations on the same line are linked
by 'ride' edges, and official line-to-line connections (including different-name
transfers such as 溜池山王 ↔ 国会議事堂前) are linked by 'transfer' edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tokyo_metro_mvp.json"


@dataclass(frozen=True)
class LineInfo:
    id: str
    name: str
    name_en: str
    color: str
    text_color: str
    forward_direction: str
    backward_direction: str
    branch_direction: Optional[str] = None


@dataclass(frozen=True)
class StationNode:
    id: str
    code: str
    station_name: str
    station_name_en: str
    line_id: str
    branch: str
    x: float
    y: float
    lx: float
    ly: float
    anchor: str
    is_major: bool = False
    hide_label: bool = False


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str  # "ride" | "transfer"
    line_id: Optional[str] = None
    direction_label: Optional[str] = None
    segment_id: Optional[str] = None
    transfer_id: Optional[str] = None
    transfer_label: Optional[str] = None
    is_diff_name: bool = False


@dataclass(frozen=True)
class TransferInfo:
    id: str
    node_a: str
    node_b: str
    label: str
    is_diff_name: bool
    note: str = ""


@dataclass(frozen=True)
class SegmentInfo:
    id: str
    node_a: str
    node_b: str
    line_id: str
    label: str


def make_segment_id(node_a: str, node_b: str) -> str:
    """Return canonical undirected segment identifier."""
    a, b = sorted((node_a, node_b))
    return f"{a}__{b}"


class MetroGraph:
    """In-memory station-line graph for Tokyo Metro."""

    def __init__(self, raw_data: dict) -> None:
        self.metadata: dict = raw_data.get("metadata", {})
        self.lines: Dict[str, LineInfo] = {
            item["id"]: LineInfo(
                id=item["id"],
                name=item["name"],
                name_en=item["name_en"],
                color=item["color"],
                text_color=item.get("text_color", "#FFFFFF"),
                forward_direction=item["forward_direction"],
                backward_direction=item["backward_direction"],
                branch_direction=item.get("branch_direction"),
            )
            for item in raw_data["lines"]
        }

        self.nodes: Dict[str, StationNode] = {}
        self.station_name_to_nodes: Dict[str, List[str]] = {}
        for item in raw_data["nodes"]:
            node = StationNode(
                id=item["id"],
                code=item["code"],
                station_name=item["station_name"],
                station_name_en=item["station_name_en"],
                line_id=item["line_id"],
                branch=item.get("branch", "main"),
                x=float(item["x"]),
                y=float(item["y"]),
                lx=float(item.get("lx", 0)),
                ly=float(item.get("ly", -14)),
                anchor=item.get("anchor", "middle"),
                is_major=bool(item.get("is_major", False)),
                hide_label=bool(item.get("hide_label", False)),
            )
            self.nodes[node.id] = node
            self.station_name_to_nodes.setdefault(node.station_name, []).append(node.id)

        self.adjacency: Dict[str, List[Edge]] = {node_id: [] for node_id in self.nodes}
        self.segments: Dict[str, SegmentInfo] = {}
        self.transfers: Dict[str, TransferInfo] = {}

        for seg in raw_data["ride_segments"]:
            u = seg["from"]
            v = seg["to"]
            line_id = seg["line_id"]
            seg_id = make_segment_id(u, v)
            line_name = self.lines[line_id].name
            label = f"{line_name}：{self.nodes[u].station_name} — {self.nodes[v].station_name}"
            self.segments[seg_id] = SegmentInfo(
                id=seg_id,
                node_a=u,
                node_b=v,
                line_id=line_id,
                label=label,
            )
            self.adjacency[u].append(
                Edge(
                    source=u,
                    target=v,
                    kind="ride",
                    line_id=line_id,
                    direction_label=seg["forward_dir"],
                    segment_id=seg_id,
                )
            )
            self.adjacency[v].append(
                Edge(
                    source=v,
                    target=u,
                    kind="ride",
                    line_id=line_id,
                    direction_label=seg["backward_dir"],
                    segment_id=seg_id,
                )
            )

        for tr in raw_data["transfers"]:
            tr_info = TransferInfo(
                id=tr["id"],
                node_a=tr["node_a"],
                node_b=tr["node_b"],
                label=tr["label"],
                is_diff_name=bool(tr.get("is_diff_name", False)),
                note=tr.get("note", ""),
            )
            self.transfers[tr_info.id] = tr_info
            for src, dst in ((tr_info.node_a, tr_info.node_b), (tr_info.node_b, tr_info.node_a)):
                self.adjacency[src].append(
                    Edge(
                        source=src,
                        target=dst,
                        kind="transfer",
                        transfer_id=tr_info.id,
                        transfer_label=tr_info.label,
                        is_diff_name=tr_info.is_diff_name,
                    )
                )

    @classmethod
    def from_json(cls, path: Path | str = DEFAULT_DATA_PATH) -> "MetroGraph":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return cls(raw_data)

    def get_station_names(self) -> List[str]:
        """Return unique station names ordered cleanly by line appearance."""
        seen: List[str] = []
        for node in self.nodes.values():
            if node.station_name not in seen:
                seen.append(node.station_name)
        return seen

    def get_station_display_label(self, station_name: str) -> str:
        """Return e.g. '銀座 [G/M]' or '浅草 [G]' for dropdown clarity."""
        node_ids = self.station_name_to_nodes[station_name]
        codes = "/".join(self.nodes[nid].code for nid in node_ids)
        return f"{station_name} ({codes})"

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        for edge in self.adjacency.get(source, []):
            if edge.target == target:
                return edge
        return None

    def get_ordered_segments(self) -> List[SegmentInfo]:
        return list(self.segments.values())

    def get_ordered_transfers(self) -> List[TransferInfo]:
        return list(self.transfers.values())

    def get_line_colors_for_station(self, station_name: str) -> List[Tuple[str, str]]:
        return [
            (self.nodes[nid].line_id, self.lines[self.nodes[nid].line_id].color)
            for nid in self.station_name_to_nodes.get(station_name, [])
        ]
