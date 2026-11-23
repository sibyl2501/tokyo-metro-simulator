"""Crisp Tokyo Metro-style SVG schematic route map renderer.

Renders:
- Ginza Line (#FF9500) and Marunouchi Line + Honancho Branch (#F62E36)
- Official transfer pills (same-name and different-name dashed connectors)
- Highlighted active route (with glow, start/end pins, and active transfer markers)
- Optional ghost comparison overlay of the baseline (before-change) route
- Clear visual badges for disabled transfers and closed track segments
"""

from __future__ import annotations

import html
from typing import Iterable, Optional, Set, Tuple

from src.metro_graph import MetroGraph
from src.routing import RouteCandidate


def render_metro_svg(
    graph: MetroGraph,
    active_route: Optional[RouteCandidate] = None,
    baseline_route: Optional[RouteCandidate] = None,
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
    origin_station: Optional[str] = None,
    destination_station: Optional[str] = None,
) -> str:
    """Return a self-contained responsive SVG string representing the route map."""
    disabled_tr: Set[str] = set(disabled_transfers or ())
    closed_seg: Set[str] = set(closed_segments or ())

    active_seg_ids: Set[str] = set(active_route.segment_ids) if active_route else set()
    active_tr_ids: Set[str] = set(active_route.transfer_ids) if active_route else set()
    active_node_ids: Set[str] = set(active_route.node_ids) if active_route else set()

    baseline_seg_ids: Set[str] = set()
    baseline_tr_ids: Set[str] = set()
    show_baseline = (
        baseline_route is not None
        and active_route is not None
        and baseline_route.signature != active_route.signature
    )
    if show_baseline and baseline_route is not None:
        baseline_seg_ids = set(baseline_route.segment_ids)
        baseline_tr_ids = set(baseline_route.transfer_ids)

    has_active = active_route is not None
    parts: list[str] = []

    parts.append(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1260 600" """
        """style="width:100%;height:auto;background:#FAFBFC;border:1px solid #E2E8F0;border-radius:14px;font-family:-apple-system,BlinkMacSystemFont,'Hiragino Sans','Yu Gothic',sans-serif;">"""
    )

    # Defs: glow filter + hatch pattern
    parts.append(
        """<defs>
  <filter id="routeGlow" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="3.5" result="blur"/>
    <feMerge>
      <feMergeNode in="blur"/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>
  <filter id="softShadow" x="-15%" y="-15%" width="130%" height="130%">
    <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#0F172A" flood-opacity="0.18"/>
  </filter>
</defs>"""
    )

    # Subtle background grid
    parts.append('<g opacity="0.35">')
    for gx in range(60, 1260, 80):
        parts.append(f'<line x1="{gx}" y1="20" x2="{gx}" y2="580" stroke="#E2E8F0" stroke-width="1"/>')
    for gy in range(60, 600, 80):
        parts.append(f'<line x1="20" y1="{gy}" x2="1240" y2="{gy}" stroke="#E2E8F0" stroke-width="1"/>')
    parts.append("</g>")

    # Map title & legend box (top-left)
    parts.append(
        """<g transform="translate(24, 20)">
  <rect x="0" y="0" width="388" height="104" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2" filter="url(#softShadow)"/>
  <text x="14" y="23" font-size="13.5" font-weight="700" fill="#0F172A">東京メトロ 路線図（銀座線・丸ノ内線）</text>
  <line x1="14" y1="43" x2="40" y2="43" stroke="#FF9500" stroke-width="5.5" stroke-linecap="round"/>
  <text x="48" y="47" font-size="11.5" font-weight="600" fill="#334155">G 銀座線</text>
  <line x1="118" y1="43" x2="144" y2="43" stroke="#F62E36" stroke-width="5.5" stroke-linecap="round"/>
  <text x="152" y="47" font-size="11.5" font-weight="600" fill="#334155">M 丸ノ内線（方南町支線含む）</text>
  <rect x="14" y="60" width="26" height="11" rx="5.5" fill="#FFFFFF" stroke="#475569" stroke-width="1.8"/>
  <text x="48" y="69.5" font-size="11" fill="#475569">公式乗換</text>
  <line x1="118" y1="65.5" x2="144" y2="65.5" stroke="#475569" stroke-width="2.5" stroke-dasharray="4 3"/>
  <text x="152" y="69.5" font-size="11" fill="#475569">異駅名乗換（連絡通路）</text>
  <circle cx="27" cy="88" r="6.5" fill="#DC2626"/>
  <text x="27" y="91.5" font-size="9" font-weight="700" fill="#FFFFFF" text-anchor="middle">✕</text>
  <text x="48" y="92" font-size="11" fill="#991B1B">乗換無効 / 区間閉鎖</text>
  <line x1="186" y1="88" x2="216" y2="88" stroke="#2563EB" stroke-width="3.5" stroke-dasharray="6 4"/>
  <text x="224" y="92" font-size="11" fill="#1D4ED8">変更前の経路</text>
</g>"""
    )

    # 1. Base track segments
    for seg in graph.segments.values():
        na = graph.nodes[seg.node_a]
        nb = graph.nodes[seg.node_b]
        color = graph.lines[seg.line_id].color
        is_closed = seg.id in closed_seg
        is_active = seg.id in active_seg_ids

        if is_closed:
            parts.append(
                f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                f'stroke="#CBD5E1" stroke-width="6" stroke-dasharray="6 6" stroke-linecap="round">'
                f"<title>{html.escape(seg.label)}（閉鎖中）</title></line>"
            )
            mx, my = (na.x + nb.x) / 2, (na.y + nb.y) / 2
            parts.append(
                f'<g><circle cx="{mx}" cy="{my}" r="10" fill="#DC2626" stroke="#FFFFFF" stroke-width="2"/>'
                f'<text x="{mx}" y="{my + 3.8}" font-size="11" font-weight="800" fill="#FFFFFF" text-anchor="middle">✕</text>'
                f"<title>区間閉鎖: {html.escape(seg.label)}</title></g>"
            )
        else:
            opacity = "1.0" if (not has_active or is_active) else "0.26"
            width = "6.5" if not has_active else ("5.5" if not is_active else "6.5")
            parts.append(
                f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" opacity="{opacity}">'
                f"<title>{html.escape(seg.label)}</title></line>"
            )

    # 2. Baseline (before-change) ghost overlay if route changed
    if show_baseline and baseline_route is not None:
        for edge in baseline_route.edges:
            na = graph.nodes[edge.source]
            nb = graph.nodes[edge.target]
            if edge.kind == "ride":
                parts.append(
                    f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                    f'stroke="#2563EB" stroke-width="4.5" stroke-dasharray="8 6" stroke-linecap="round" opacity="0.65">'
                    f"<title>変更前の経路: {html.escape(baseline_route.title)}</title></line>"
                )
            elif edge.kind == "transfer":
                parts.append(
                    f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                    f'stroke="#2563EB" stroke-width="3.5" stroke-dasharray="4 4" opacity="0.7"/>'
                )

    # 3. Active route highlight tracks
    if active_route is not None:
        for edge in active_route.edges:
            if edge.kind != "ride":
                continue
            na = graph.nodes[edge.source]
            nb = graph.nodes[edge.target]
            color = graph.lines[edge.line_id].color if edge.line_id else "#0F172A"
            # Outer white halo + colored glow bar
            parts.append(
                f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                f'stroke="#FFFFFF" stroke-width="14" stroke-linecap="round" opacity="0.9"/>'
            )
            parts.append(
                f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                f'stroke="{color}" stroke-width="9.5" stroke-linecap="round" filter="url(#routeGlow)"/>'
            )

    # 4. Transfer connectors
    for tr in graph.transfers.values():
        na = graph.nodes[tr.node_a]
        nb = graph.nodes[tr.node_b]
        is_disabled = tr.id in disabled_tr
        is_active = tr.id in active_tr_ids
        mx, my = (na.x + nb.x) / 2, (na.y + nb.y) / 2

        if tr.is_diff_name:
            stroke = "#DC2626" if is_disabled else ("#0F172A" if is_active else "#64748B")
            sw = "4.5" if is_active else "3.0"
            parts.append(
                f'<line x1="{na.x}" y1="{na.y}" x2="{nb.x}" y2="{nb.y}" '
                f'stroke="{stroke}" stroke-width="{sw}" stroke-dasharray="5 4" '
                f'opacity="{"0.95" if (is_active or is_disabled or not has_active) else "0.45"}">'
                f"<title>異駅名公式乗換: {html.escape(tr.label)}（{html.escape(tr.note)}）</title></line>"
            )
        else:
            # Capsule connecting the two vertically stacked nodes
            top_y = min(na.y, nb.y) - 11
            height = abs(nb.y - na.y) + 22
            x_left = na.x - 11
            fill = "#FEF2F2" if is_disabled else ("#FEF9C3" if is_active else "#FFFFFF")
            stroke = "#DC2626" if is_disabled else ("#0F172A" if is_active else "#475569")
            sw = "3" if (is_active or is_disabled) else "2"
            parts.append(
                f'<rect x="{x_left}" y="{top_y}" width="22" height="{height}" rx="11" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}">'
                f"<title>乗換駅: {html.escape(tr.label)}（{html.escape(tr.note)}）</title></rect>"
            )

        if is_disabled:
            badge_x = mx + (24 if not tr.is_diff_name else 0)
            parts.append(
                f'<g><circle cx="{badge_x}" cy="{my}" r="10.5" fill="#DC2626" stroke="#FFFFFF" stroke-width="2"/>'
                f'<text x="{badge_x}" y="{my + 3.8}" font-size="11" font-weight="800" fill="#FFFFFF" text-anchor="middle">✕</text>'
                f"<title>乗換無効: {html.escape(tr.label)}（通過列車は利用可）</title></g>"
            )
        elif is_active:
            badge_x = mx + (26 if not tr.is_diff_name else 22)
            parts.append(
                f'<g filter="url(#softShadow)">'
                f'<rect x="{badge_x - 19}" y="{my - 10}" width="38" height="20" rx="10" fill="#0F172A"/>'
                f'<text x="{badge_x}" y="{my + 3.8}" font-size="10.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">乗換</text>'
                f"</g>"
            )

    # 5. Station nodes & labels
    origin_nodes = set(graph.station_name_to_nodes.get(origin_station or "", []))
    dest_nodes = set(graph.station_name_to_nodes.get(destination_station or "", []))

    for node in graph.nodes.values():
        color = graph.lines[node.line_id].color
        is_in_active = node.id in active_node_ids
        is_origin_used = (
            active_route is not None and active_route.node_ids[0] == node.id
        ) or (active_route is None and node.id in origin_nodes)
        is_dest_used = (
            active_route is not None and active_route.node_ids[-1] == node.id
        ) or (active_route is None and node.id in dest_nodes)

        r = 7.5 if (node.is_major or is_in_active or is_origin_used or is_dest_used) else 5.5
        opacity = "1.0" if (not has_active or is_in_active or node.station_name in (origin_station, destination_station)) else "0.42"
        stroke_w = "3.2" if is_in_active else "2.4"

        parts.append(
            f'<circle cx="{node.x}" cy="{node.y}" r="{r}" fill="#FFFFFF" '
            f'stroke="{color}" stroke-width="{stroke_w}" opacity="{opacity}">'
            f"<title>{html.escape(node.station_name)} ({node.code} / {graph.lines[node.line_id].name})</title></circle>"
        )

        if not node.hide_label:
            label_opacity = "1.0" if (not has_active or is_in_active or node.is_major or node.station_name in (origin_station, destination_station)) else "0.48"
            font_weight = "700" if (node.is_major or is_in_active or node.station_name in (origin_station, destination_station)) else "500"
            font_size = "12.5" if (node.is_major or is_origin_used or is_dest_used) else "10.8"
            fill = "#0F172A" if (node.is_major or is_in_active) else "#334155"
            parts.append(
                f'<text x="{node.x + node.lx}" y="{node.y + node.ly}" '
                f'font-size="{font_size}" font-weight="{font_weight}" fill="{fill}" '
                f'text-anchor="{node.anchor}" opacity="{label_opacity}">'
                f"{html.escape(node.station_name)}</text>"
            )

    # 6. Origin ("発") and Destination ("着") callout badges
    def _draw_pin(node_id: str, tag: str, bg_color: str) -> None:
        node = graph.nodes[node_id]
        py = node.y - 24
        parts.append(
            f'<g filter="url(#softShadow)">'
            f'<rect x="{node.x - 16}" y="{py - 12}" width="32" height="20" rx="6" fill="{bg_color}"/>'
            f'<polygon points="{node.x - 5},{py + 8} {node.x + 5},{py + 8} {node.x},{py + 14}" fill="{bg_color}"/>'
            f'<text x="{node.x}" y="{py + 2}" font-size="11.5" font-weight="800" fill="#FFFFFF" text-anchor="middle">{tag}</text>'
            f"</g>"
        )

    if active_route is not None:
        _draw_pin(active_route.node_ids[0], "発", "#059669")
        if active_route.node_ids[-1] != active_route.node_ids[0]:
            _draw_pin(active_route.node_ids[-1], "着", "#1D4ED8")
    else:
        if origin_station and origin_station in graph.station_name_to_nodes:
            _draw_pin(graph.station_name_to_nodes[origin_station][0], "発", "#059669")
        if destination_station and destination_station in graph.station_name_to_nodes and destination_station != origin_station:
            _draw_pin(graph.station_name_to_nodes[destination_station][0], "着", "#1D4ED8")

    # Branch label helper text
    parts.append(
        '<text x="155" y="405" font-size="11" font-weight="600" fill="#DC2626" opacity="0.85" text-anchor="end">丸ノ内線</text>'
    )
    parts.append(
        '<text x="155" y="420" font-size="10.5" fill="#DC2626" opacity="0.85" text-anchor="end">方南町支線</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)
