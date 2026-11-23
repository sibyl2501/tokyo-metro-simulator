"""東京メトロ 乗換シミュレーター（MVP: 銀座線＋丸ノ内線）のユニットテスト集.

グラフ構造の整合性、基本の経路探索、方南町支線、乗換無効化・区間閉鎖の制約、
SVG 路線図の生成、および PyTorch 好み学習モデルによる再ランキングの変化を自動検証します。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.map_renderer import render_metro_svg
from src.metro_graph import MetroGraph
from src.preference_model import (
    generate_sample_choice_logs,
    rerank_candidates,
    train_preference_model,
)
from src.routing import search_routes


@pytest.fixture(scope="module")
def graph() -> MetroGraph:
    """テストモジュール全体で共有する MetroGraph フィクスチャ（JSON 読み込みを 1 回にまとめる）."""
    return MetroGraph.from_json()


def test_graph_structure(graph: MetroGraph) -> None:
    """路線網のノード数・ユニーク駅名数・公式乗換ポイントの定義が正しいかを検証する."""
    # 銀座線 19 駅 + 丸ノ内線本線 25 駅 + 方南町支線 3 駅 = 計 47 の「駅×路線」ノード
    assert len(graph.nodes) == 47
    # 赤坂見附・銀座の 2 駅は 2 路線重複するため、駅名ベースでは 47 - 2 = 45 駅
    assert len(graph.get_station_names()) == 45
    assert set(graph.transfers.keys()) == {
        "transfer:赤坂見附",
        "transfer:溜池山王-国会議事堂前",
        "transfer:銀座",
    }
    # 溜池山王 ↔ 国会議事堂前は異駅名の公式乗換フラグが立っていること
    assert graph.transfers["transfer:溜池山王-国会議事堂前"].is_diff_name is True


def test_same_line_zero_transfers(graph: MetroGraph) -> None:
    """同一路線の端から端までの移動が、乗換 0 回・正確な駅数で探索されることを検証する."""
    # 銀座線：浅草 -> 渋谷（全 19 駅 = 18 区間）
    g_routes = search_routes(graph, "浅草", "渋谷", priority="fewest_transfers")
    assert g_routes
    assert g_routes[0].transfers == 0
    assert g_routes[0].stops == 18
    assert "銀座線" in g_routes[0].guidance

    # 丸ノ内線本線：池袋 -> 荻窪（全 25 駅 = 24 区間）
    m_routes = search_routes(graph, "池袋", "荻窪", priority="fewest_transfers")
    assert m_routes
    assert m_routes[0].transfers == 0
    assert m_routes[0].stops == 24
    assert "丸ノ内線" in m_routes[0].guidance


def test_marunouchi_honancho_branch(graph: MetroGraph) -> None:
    """丸ノ内線・方南町支線からの直通（新宿・銀座）および中野坂上折り返し（荻窪）を検証する."""
    # 方南町 -> 新宿：方南町支線から本線へ直通（乗換 0 回・5 区間）
    r_shinjuku = search_routes(graph, "方南町", "新宿", priority="fewest_transfers")
    assert r_shinjuku
    assert r_shinjuku[0].transfers == 0
    assert r_shinjuku[0].stops == 5

    # 方南町 -> 銀座：丸ノ内線直通（乗換 0 回・13 区間）
    r_ginza = search_routes(graph, "方南町", "銀座", priority="fewest_transfers")
    assert r_ginza
    assert r_ginza[0].transfers == 0
    assert r_ginza[0].stops == 13

    # 方南町 -> 荻窪：同じ丸ノ内線なので路線乗換は 0 回だが、中野坂上で方面切替のため leg は 2 分割される
    r_ogikubo = search_routes(graph, "方南町", "荻窪", priority="fewest_transfers")
    assert r_ogikubo
    assert r_ogikubo[0].transfers == 0
    assert r_ogikubo[0].stops == 8
    assert len(r_ogikubo[0].legs) == 2
    assert "中野坂上" in r_ogikubo[0].guidance


def test_asakusa_to_shinjuku_and_disable_ginza_transfer(graph: MetroGraph) -> None:
    """浅草→新宿の通常候補と、「銀座で乗換無効」にしたときの代替探索・通過可否を検証する."""
    # 通常検索：銀座経由 (18 区間) が 1 位、赤坂見附経由・溜池山王経由 (各 19 区間) も候補に挙がる
    before = search_routes(graph, "浅草", "新宿", priority="fewest_transfers", max_candidates=5)
    titles_before = [c.title for c in before]
    assert "銀座経由" in titles_before
    assert "赤坂見附経由" in titles_before
    assert "溜池山王・国会議事堂前経由" in titles_before
    assert before[0].title == "銀座経由"
    assert before[0].transfers == 1
    assert before[0].stops == 18

    # 銀座での乗換を無効化：銀座経由だけが消え、赤坂見附経由・溜池山王経由が代替候補として残る
    after = search_routes(
        graph,
        "浅草",
        "新宿",
        priority="fewest_transfers",
        max_candidates=5,
        disabled_transfers=["transfer:銀座"],
    )
    titles_after = [c.title for c in after]
    assert "銀座経由" not in titles_after
    assert "赤坂見附経由" in titles_after
    assert "溜池山王・国会議事堂前経由" in titles_after

    # 重要要件：銀座の乗換を無効にしても、銀座駅を「通過するだけ」の銀座線列車（新橋 -> 京橋）は利用可能であること
    pass_through = search_routes(
        graph,
        "新橋",
        "京橋",
        priority="fewest_transfers",
        disabled_transfers=["transfer:銀座"],
    )
    assert pass_through
    assert pass_through[0].transfers == 0
    assert pass_through[0].stops == 2
    assert "G-09" in pass_through[0].node_ids

    # 3 つすべての公式乗換を無効にすると銀座線と丸ノ内線が分断され、到達不可（空リスト）になる
    blocked_all = search_routes(
        graph,
        "浅草",
        "新宿",
        disabled_transfers=[
            "transfer:銀座",
            "transfer:赤坂見附",
            "transfer:溜池山王-国会議事堂前",
        ],
    )
    assert blocked_all == []


def test_segment_closure_detour(graph: MetroGraph) -> None:
    """区間閉鎖モードで指定区間が通れないときに、別路線を使った迂回経路が見つかることを検証する."""
    # 通常の新宿 -> 東京は丸ノ内線直通（乗換 0 回・9 区間）
    normal = search_routes(graph, "新宿", "東京", priority="fewest_transfers")
    assert normal[0].transfers == 0
    assert normal[0].stops == 9

    # 丸ノ内線の国会議事堂前 (M-14) — 霞ケ関 (M-15) 間を閉鎖 -> 銀座線へ迂回する 2 回乗換ルートへ切替
    detour = search_routes(
        graph,
        "新宿",
        "東京",
        priority="fewest_transfers",
        closed_segments=["M-14__M-15"],
    )
    assert detour
    assert detour[0].transfers == 2  # 丸ノ内線 -> 銀座線 -> 丸ノ内線 の迂回
    assert "M-14__M-15" not in detour[0].segment_ids


def test_priority_tradeoff_shibuya_to_ginza(graph: MetroGraph) -> None:
    """渋谷→銀座で「乗換 0 回・8 区間（銀座線直通）」と「乗換 1 回・7 区間（赤坂見附経由）」の優先切替を検証する."""
    by_transfers = search_routes(graph, "渋谷", "銀座", priority="fewest_transfers")
    by_stops = search_routes(graph, "渋谷", "銀座", priority="fewest_stops")

    assert by_transfers[0].transfers == 0
    assert by_transfers[0].stops == 8

    assert by_stops[0].stops == 7
    assert by_stops[0].transfers == 1


def test_svg_map_rendering(graph: MetroGraph) -> None:
    """変更前後ルート・無効化乗換を含む SVG 路線図が正しく生成されることを検証する."""
    routes = search_routes(graph, "浅草", "新宿", disabled_transfers=["transfer:銀座"])
    baseline = search_routes(graph, "浅草", "新宿")
    svg = render_metro_svg(
        graph=graph,
        active_route=routes[0],
        baseline_route=baseline[0],
        disabled_transfers=["transfer:銀座"],
        origin_station="浅草",
        destination_station="新宿",
    )
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "浅草" in svg
    assert "新宿" in svg


def test_pytorch_preference_learning_personas(graph: MetroGraph, tmp_path: Path) -> None:
    """PyTorch 好み学習が 2 種類のペルソナ（乗換回避派 vs 駅数最短派）を正しく学習し、順位を分岐させることを検証する."""
    # ペルソナ 1: 乗換回避派 -> Loss が減少し、Top-1 一致率 85% 以上、許容遠回り駅数が 1.5 駅超になること
    avoid_logs = generate_sample_choice_logs(graph, persona="avoid_transfers", log_path=None)
    model_avoid, summary_avoid = train_preference_model(
        avoid_logs, epochs=250, save_path=tmp_path / "avoid.pt"
    )
    assert summary_avoid.final_loss < summary_avoid.initial_loss
    assert summary_avoid.top1_accuracy >= 0.85
    assert summary_avoid.transfer_equivalent_stops is not None
    assert summary_avoid.transfer_equivalent_stops > 1.5

    # ペルソナ 2: 駅数最短派 -> Loss が減少し、Top-1 一致率 85% 以上になること
    short_logs = generate_sample_choice_logs(graph, persona="shortest_stops", log_path=None)
    model_short, summary_short = train_preference_model(
        short_logs, epochs=250, save_path=tmp_path / "short.pt"
    )
    assert summary_short.final_loss < summary_short.initial_loss
    assert summary_short.top1_accuracy >= 0.85

    # 渋谷→銀座（0 回/8 区間 vs 1 回/7 区間）で、2 つのモデルの 1 位予測が意図通りに分かれることを確認
    cands = search_routes(graph, "渋谷", "銀座", priority="fewest_transfers")
    top_avoid = rerank_candidates(cands, model_avoid)[0]
    top_short = rerank_candidates(cands, model_short)[0]
    assert top_avoid.transfers == 0
    assert top_short.stops == 7
