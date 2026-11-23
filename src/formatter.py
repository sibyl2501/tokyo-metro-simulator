"""Natural Japanese route titles and step-by-step guidance formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.routing import RouteCandidate


def format_route_title(candidate: "RouteCandidate") -> str:
    """Create a concise candidate label such as '銀座経由' or '銀座線 直通'."""
    if candidate.origin_station == candidate.destination_station:
        return "同一駅"

    if candidate.transfers == 0:
        if not candidate.legs:
            return "直通"
        first_line = candidate.legs[0].line_name
        if len(candidate.legs) > 1:
            return f"{first_line}（中野坂上で方面切替）"
        if candidate.legs[0].is_branch_leg:
            return f"{first_line}（方南町支線）直通"
        return f"{first_line} 直通"

    via_names = []
    for step in candidate.transfer_steps:
        if step.is_diff_name:
            via_names.append(f"{step.from_station}・{step.to_station}")
        else:
            via_names.append(step.from_station)
    return "・".join(via_names) + "経由"


def format_route_guidance(candidate: "RouteCandidate") -> str:
    """Create human-readable Japanese guidance matching the design spec.

    Example:
      '浅草から銀座線・渋谷方面へ。銀座で丸ノ内線・荻窪方面に乗り換えて新宿へ。'
    """
    if candidate.origin_station == candidate.destination_station:
        return f"{candidate.origin_station}から移動はありません。"

    # Special case: pure walking transfer between two adjacent different-name stations (e.g. 溜池山王 -> 国会議事堂前)
    if not candidate.legs and candidate.transfer_steps:
        step = candidate.transfer_steps[0]
        return f"{step.from_station}から改札内連絡通路で{step.to_station}へ移動。"

    parts = []
    # Map transfers by the leg index they occur after.
    # Note: a branch turn at Nakano-sakaue produces two legs without a line transfer.
    leg_idx = 0
    tr_idx = 0
    total_legs = len(candidate.legs)

    while leg_idx < total_legs:
        leg = candidate.legs[leg_idx]
        start_st = leg.station_names[0]
        end_st = leg.station_names[-1]
        line_str = f"{leg.line_name}・{leg.direction_label}" if leg.direction_label else leg.line_name

        if leg_idx == 0:
            if total_legs == 1 and tr_idx >= len(candidate.transfer_steps):
                parts.append(f"{start_st}から{line_str}へ乗り、{end_st}へ。")
            else:
                parts.append(f"{start_st}から{line_str}へ。")
        else:
            prev_leg = candidate.legs[leg_idx - 1]
            # Check if there is a transfer step connecting prev_leg -> leg
            if (
                tr_idx < len(candidate.transfer_steps)
                and candidate.transfer_steps[tr_idx].from_node_id == prev_leg.node_ids[-1]
                and candidate.transfer_steps[tr_idx].to_node_id == leg.node_ids[0]
            ):
                step = candidate.transfer_steps[tr_idx]
                tr_idx += 1
                is_last = leg_idx == total_legs - 1
                if step.is_diff_name:
                    if is_last:
                        parts.append(
                            f"{step.from_station}から連絡通路で{step.to_station}へ移動し、"
                            f"{line_str}に乗り換えて{end_st}へ。"
                        )
                    else:
                        parts.append(
                            f"{step.from_station}から連絡通路で{step.to_station}へ移動し、"
                            f"{line_str}へ乗り換え。"
                        )
                else:
                    if is_last:
                        parts.append(f"{step.from_station}で{line_str}に乗り換えて{end_st}へ。")
                    else:
                        parts.append(f"{step.from_station}で{line_str}へ乗り換え。")
            else:
                # Same-line branch turn at Nakano-sakaue
                is_last = leg_idx == total_legs - 1
                if is_last:
                    parts.append(f"{start_st}で{line_str}へ乗り継いで{end_st}へ。")
                else:
                    parts.append(f"{start_st}で{line_str}へ乗り継ぎ。")
        leg_idx += 1

    # Handle trailing transfer (rare, e.g., arriving via different-name transfer)
    while tr_idx < len(candidate.transfer_steps):
        step = candidate.transfer_steps[tr_idx]
        tr_idx += 1
        if step.is_diff_name:
            parts.append(f"{step.from_station}から連絡通路で{step.to_station}へ。")

    return "".join(parts)
