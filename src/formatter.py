"""経路候補の見出しタイトルと自然な日本語案内文を生成するフォーマッターモジュール.

探索エンジン（src/routing.py）が構築した RouteCandidate の乗車区間列（legs）と
乗換ステップ列（transfer_steps）をもとに、以下の 2 種類のテキストを生成します。
  1. format_route_title   : 比較表やカード見出し用の短いラベル（例: "銀座経由", "銀座線 直通"）
  2. format_route_guidance: 乗換案内アプリ風の自然な日本語説明文
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# 実行時の循環インポートを避けつつ、型チェック時だけ RouteCandidate を参照する
if TYPE_CHECKING:
    from src.routing import RouteCandidate


def format_route_title(candidate: "RouteCandidate") -> str:
    """経路候補の簡潔な見出しラベルを生成する（例: '銀座経由', '銀座線 直通'）."""
    if candidate.origin_station == candidate.destination_station:
        return "同一駅"

    # 路線間の乗換がない場合（直通、または丸ノ内線・中野坂上での支線折り返し）
    if candidate.transfers == 0:
        if not candidate.legs:
            return "直通"
        first_line = candidate.legs[0].line_name
        if len(candidate.legs) > 1:
            return f"{first_line}（中野坂上で方面切替）"
        if candidate.legs[0].is_branch_leg:
            return f"{first_line}（方南町支線）直通"
        return f"{first_line} 直通"

    # 乗換がある場合は経由する乗換駅名を「・」でつないで「〇〇経由」にする
    via_names = []
    for step in candidate.transfer_steps:
        if step.is_diff_name:
            # 異駅名乗換は両方の駅名を明示する（例: 溜池山王・国会議事堂前経由）
            via_names.append(f"{step.from_station}・{step.to_station}")
        else:
            via_names.append(step.from_station)
    return "・".join(via_names) + "経由"


def format_route_guidance(candidate: "RouteCandidate") -> str:
    """人間がそのまま読んで行動できる自然な日本語の乗換案内文を生成する.

    出力例:
      '浅草から銀座線・渋谷方面へ。銀座で丸ノ内線・荻窪方面に乗り換えて新宿へ。'
    """
    if candidate.origin_station == candidate.destination_station:
        return f"{candidate.origin_station}から移動はありません。"

    # 特殊ケース: 電車に乗らず、異駅名の改札内連絡通路だけを歩いて移動する場合（例: 溜池山王 -> 国会議事堂前）
    if not candidate.legs and candidate.transfer_steps:
        step = candidate.transfer_steps[0]
        return f"{step.from_station}から改札内連絡通路で{step.to_station}へ移動。"

    parts = []
    # 乗車区間 (legs) と乗換 (transfer_steps) を先頭から順に突き合わせながら文章を組み立てる。
    # 注意: 中野坂上での支線折り返しでは、乗換ステップを挟まずに 2 つの leg が連続する。
    leg_idx = 0
    tr_idx = 0
    total_legs = len(candidate.legs)

    while leg_idx < total_legs:
        leg = candidate.legs[leg_idx]
        start_st = leg.station_names[0]
        end_st = leg.station_names[-1]
        line_str = f"{leg.line_name}・{leg.direction_label}" if leg.direction_label else leg.line_name

        if leg_idx == 0:
            # 最初の乗車区間（直通 1 本で完結する場合と、後ろにまだ乗換・乗り継ぎが続く場合で文末を分ける）
            if total_legs == 1 and tr_idx >= len(candidate.transfer_steps):
                parts.append(f"{start_st}から{line_str}へ乗り、{end_st}へ。")
            else:
                parts.append(f"{start_st}から{line_str}へ。")
        else:
            prev_leg = candidate.legs[leg_idx - 1]
            # 直前の leg の終点ノードと今回の leg の始点ノードを繋ぐ乗換ステップがあるか確認
            if (
                tr_idx < len(candidate.transfer_steps)
                and candidate.transfer_steps[tr_idx].from_node_id == prev_leg.node_ids[-1]
                and candidate.transfer_steps[tr_idx].to_node_id == leg.node_ids[0]
            ):
                step = candidate.transfer_steps[tr_idx]
                tr_idx += 1
                is_last = leg_idx == total_legs - 1
                if step.is_diff_name:
                    # 異駅名乗換（連絡通路移動を明記）
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
                    # 同一駅名での通常乗換
                    if is_last:
                        parts.append(f"{step.from_station}で{line_str}に乗り換えて{end_st}へ。")
                    else:
                        parts.append(f"{step.from_station}で{line_str}へ乗り換え。")
            else:
                # 乗換ステップなしに次の leg へ移るケース = 中野坂上での同路線・支線乗り継ぎ
                is_last = leg_idx == total_legs - 1
                if is_last:
                    parts.append(f"{start_st}で{line_str}へ乗り継いで{end_st}へ。")
                else:
                    parts.append(f"{start_st}で{line_str}へ乗り継ぎ。")
        leg_idx += 1

    # 末尾に連絡通路の徒歩移動だけが残っているレアケース（例: 溜池山王降車後に国会議事堂前へ到着）
    while tr_idx < len(candidate.transfer_steps):
        step = candidate.transfer_steps[tr_idx]
        tr_idx += 1
        if step.is_diff_name:
            parts.append(f"{step.from_station}から連絡通路で{step.to_station}へ。")

    return "".join(parts)
