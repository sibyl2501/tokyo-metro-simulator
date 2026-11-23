"""Unit tests for the Tokyo Metro Transfer Simulator MVP (Ginza + Marunouchi)."""

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
    return MetroGraph.from_json()


def test_graph_structure(graph: MetroGraph) -> None:
    # 19 Ginza + 25 Marunouchi Main + 3 Honancho Branch = 47 station-line nodes
    assert len(graph.nodes) == 47
    # 47 - 2 shared names (赤坂見附, 銀座) = 45 unique station names
    assert len(graph.get_station_names()) == 45
    assert set(graph.transfers.keys()) == {
        "transfer:赤坂見附",
        "transfer:溜池山王-国会議事堂前",
        "transfer:銀座",
    }
    assert graph.transfers["transfer:溜池山王-国会議事堂前"].is_diff_name is True


def test_same_line_zero_transfers(graph: MetroGraph) -> None:
    # Ginza Line end-to-end
    g_routes = search_routes(graph, "浅草", "渋谷", priority="fewest_transfers")
    assert g_routes
    assert g_routes[0].transfers == 0
    assert g_routes[0].stops == 18
    assert "銀座線" in g_routes[0].guidance

    # Marunouchi Main end-to-end
    m_routes = search_routes(graph, "池袋", "荻窪", priority="fewest_transfers")
    assert m_routes
    assert m_routes[0].transfers == 0
    assert m_routes[0].stops == 24
    assert "丸ノ内線" in m_routes[0].guidance


def test_marunouchi_honancho_branch(graph: MetroGraph) -> None:
    # Honancho -> Shinjuku: direct on Marunouchi Line (0 transfers, 5 stops)
    r_shinjuku = search_routes(graph, "方南町", "新宿", priority="fewest_transfers")
    assert r_shinjuku
    assert r_shinjuku[0].transfers == 0
    assert r_shinjuku[0].stops == 5

    # Honancho -> Ginza: direct on Marunouchi Line (0 transfers, 13 stops)
    r_ginza = search_routes(graph, "方南町", "銀座", priority="fewest_transfers")
    assert r_ginza
    assert r_ginza[0].transfers == 0
    assert r_ginza[0].stops == 13

    # Honancho -> Ogikubo: connects through Nakano-sakaue (0 line transfers, 8 stops, 2 legs)
    r_ogikubo = search_routes(graph, "方南町", "荻窪", priority="fewest_transfers")
    assert r_ogikubo
    assert r_ogikubo[0].transfers == 0
    assert r_ogikubo[0].stops == 8
    assert len(r_ogikubo[0].legs) == 2
    assert "中野坂上" in r_ogikubo[0].guidance


def test_asakusa_to_shinjuku_and_disable_ginza_transfer(graph: MetroGraph) -> None:
    # Default search: Ginza (18 stops), Akasaka-mitsuke (19 stops), Tameike-sanno/Kokkai-gijidomae (19 stops)
    before = search_routes(graph, "浅草", "新宿", priority="fewest_transfers", max_candidates=5)
    titles_before = [c.title for c in before]
    assert "銀座経由" in titles_before
    assert "赤坂見附経由" in titles_before
    assert "溜池山王・国会議事堂前経由" in titles_before
    assert before[0].title == "銀座経由"
    assert before[0].transfers == 1
    assert before[0].stops == 18

    # Disable transfer at Ginza: Ginza transfer candidate disappears, Akasaka-mitsuke / Tameike-sanno remain
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

    # Crucial requirement: disabling transfer at Ginza MUST keep trains passing through Ginza usable!
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

    # Disabling all 3 transfers isolates Ginza Line from Marunouchi Line
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
    # Normal Shinjuku -> Tokyo is Marunouchi direct (0 transfers, 9 stops)
    normal = search_routes(graph, "新宿", "東京", priority="fewest_transfers")
    assert normal[0].transfers == 0
    assert normal[0].stops == 9

    # Close Marunouchi between Kokkai-gijidomae (M-14) and Kasumigaseki (M-15)
    detour = search_routes(
        graph,
        "新宿",
        "東京",
        priority="fewest_transfers",
        closed_segments=["M-14__M-15"],
    )
    assert detour
    assert detour[0].transfers == 2  # M -> G -> M detour around closed segment
    assert "M-14__M-15" not in detour[0].segment_ids


def test_priority_tradeoff_shibuya_to_ginza(graph: MetroGraph) -> None:
    by_transfers = search_routes(graph, "渋谷", "銀座", priority="fewest_transfers")
    by_stops = search_routes(graph, "渋谷", "銀座", priority="fewest_stops")

    assert by_transfers[0].transfers == 0
    assert by_transfers[0].stops == 8

    assert by_stops[0].stops == 7
    assert by_stops[0].transfers == 1


def test_svg_map_rendering(graph: MetroGraph) -> None:
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
    # Persona 1: avoid transfers
    avoid_logs = generate_sample_choice_logs(graph, persona="avoid_transfers", log_path=None)
    model_avoid, summary_avoid = train_preference_model(
        avoid_logs, epochs=250, save_path=tmp_path / "avoid.pt"
    )
    assert summary_avoid.final_loss < summary_avoid.initial_loss
    assert summary_avoid.top1_accuracy >= 0.85
    assert summary_avoid.transfer_equivalent_stops is not None
    assert summary_avoid.transfer_equivalent_stops > 1.5

    # Persona 2: shortest stops
    short_logs = generate_sample_choice_logs(graph, persona="shortest_stops", log_path=None)
    model_short, summary_short = train_preference_model(
        short_logs, epochs=250, save_path=tmp_path / "short.pt"
    )
    assert summary_short.final_loss < summary_short.initial_loss
    assert summary_short.top1_accuracy >= 0.85

    # Verify reranking actually diverges on Shibuya -> Ginza (0 transfers/8 stops vs 1 transfer/7 stops)
    cands = search_routes(graph, "渋谷", "銀座", priority="fewest_transfers")
    top_avoid = rerank_candidates(cands, model_avoid)[0]
    top_short = rerank_candidates(cands, model_short)[0]
    assert top_avoid.transfers == 0
    assert top_short.stops == 7
