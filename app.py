"""FastAPI Web アプリ本体: 東京メトロ 乗換シミュレーター（MVP: 銀座線＋丸ノ内線）.

Streamlit 版から FastAPI ＋ HTML/CSS/JS 構成へ完全移行したメインサーバーです。
既存のコアモジュール（src/metro_graph.py, src/routing.py, src/preference_model.py,
src/formatter.py, src/map_renderer.py）をそのまま呼び出し、JSON API として提供します。

【この構成のメリット】
  - 駅選択や制約の切り替えのたびに画面全体を再実行せず、fetch API で路線図と候補だけを瞬時に更新
  - `python app.py` または `uvicorn app:app --reload` のどちらでも起動可能
  - `/docs` を開けば FastAPI 自動生成の Swagger UI から各 API を直接テスト可能
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.map_renderer import render_metro_svg
from src.metro_graph import MetroGraph
from src.preference_model import (
    DEFAULT_LOG_PATH,
    DEFAULT_MODEL_PATH,
    append_choice_log,
    clear_choice_logs,
    generate_sample_choice_logs,
    load_choice_logs,
    load_saved_model,
    rerank_candidates,
    train_preference_model,
)
from src.routing import RouteCandidate, search_routes


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# FastAPI アプリケーション本体
app = FastAPI(
    title="東京メトロ 乗換シミュレーター API",
    description="銀座線・丸ノ内線（方南町支線含む）の経路探索・障害シミュレーション・PyTorch 好み学習 API",
    version="2.0.0",
)

# 路線グラフは起動時に 1 回だけメモリへ読み込んで全リクエストで共有する
graph = MetroGraph.from_json()

# 選択ログ・学習済みモデルの保存パスと直近の Loss 推移を app.state で管理
# （テスト時はここを tmp_path に差し替えることで本番データを汚さずに API テストできる）
app.state.log_path = DEFAULT_LOG_PATH
app.state.model_path = DEFAULT_MODEL_PATH
app.state.last_loss_history: List[float] = []

# フロントエンドの静的ファイル（CSS / JS）を /static 配下で配信
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# リクエストスキーマ（Pydantic による自動バリデーション）
# ==============================================================================
class SearchRequest(BaseModel):
    """経路検索および SVG 路線図生成のリクエストボディ."""

    origin: str = Field(..., description="出発駅名（例: '浅草'）")
    destination: str = Field(..., description="目的駅名（例: '新宿'）")
    priority: Literal["fewest_transfers", "fewest_stops", "learned"] = Field(
        default="fewest_transfers",
        description="優先条件（乗換少ない順 / 駅数少ない順 / PyTorch好み学習順）",
    )
    disabled_transfers: List[str] = Field(
        default_factory=list,
        description="乗換無効化する公式乗換 ID リスト",
    )
    closed_segments: List[str] = Field(
        default_factory=list,
        description="閉鎖する隣駅区間 ID リスト",
    )
    selected_index: int = Field(
        default=0,
        ge=0,
        description="路線図上でハイライト表示する候補の 0 始まりインデックス",
    )


class ChoiceLogRequest(BaseModel):
    """「👍 私ならこの経路を選ぶ」ボタン押下時の選択ログ記録リクエスト."""

    origin: str
    destination: str
    priority: Literal["fewest_transfers", "fewest_stops", "learned"] = "fewest_transfers"
    disabled_transfers: List[str] = Field(default_factory=list)
    closed_segments: List[str] = Field(default_factory=list)
    chosen_index: int = Field(..., ge=0)


class SampleLogRequest(BaseModel):
    """デモ・検証用サンプル選択ログ生成リクエスト."""

    persona: Literal["avoid_transfers", "shortest_stops"] = "avoid_transfers"


class TrainRequest(BaseModel):
    """PyTorch 好み学習のハイパーパラメータ設定."""

    epochs: int = Field(default=250, ge=10, le=2000)
    lr: float = Field(default=0.08, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.02, ge=0.0, le=1.0)


# ==============================================================================
# 内部ヘルパー関数
# ==============================================================================
def _serialize_candidate(cand: RouteCandidate) -> Dict[str, Any]:
    """RouteCandidate をフロントエンド描画用の JSON 辞書へ変換する."""
    return {
        "signature": cand.signature,
        "title": cand.title,
        "guidance": cand.guidance,
        "stops": cand.stops,
        "transfers": cand.transfers,
        "diff_name_transfers": cand.diff_name_transfers,
        "transfer_timing": round(cand.transfer_timing, 4),
        "score": cand.score,
        "legs": [
            {
                "line_id": leg.line_id,
                "line_name": leg.line_name,
                "direction_label": leg.direction_label,
                "station_names": leg.station_names,
                "stops": leg.stops,
                "is_branch_leg": leg.is_branch_leg,
            }
            for leg in cand.legs
        ],
        "transfer_steps": [
            {
                "transfer_id": step.transfer_id,
                "transfer_label": step.transfer_label,
                "from_station": step.from_station,
                "to_station": step.to_station,
                "is_diff_name": step.is_diff_name,
            }
            for step in cand.transfer_steps
        ],
    }


def _run_search_pair(
    origin: str,
    destination: str,
    priority: str,
    disabled_transfers: List[str],
    closed_segments: List[str],
) -> tuple[List[RouteCandidate], List[RouteCandidate], bool]:
    """制約なし（baseline）と制約あり（active）の両方の経路候補を計算する.

    Returns:
        (baseline_candidates, active_candidates, has_trained_model)
    """
    if origin not in graph.station_name_to_nodes:
        raise HTTPException(status_code=400, detail=f"存在しない出発駅です: {origin}")
    if destination not in graph.station_name_to_nodes:
        raise HTTPException(status_code=400, detail=f"存在しない目的駅です: {destination}")

    bundle = load_saved_model(app.state.model_path)
    trained_model = bundle[0] if bundle else None
    has_trained_model = trained_model is not None

    base_priority = "fewest_transfers" if priority == "learned" else priority
    baseline_candidates = search_routes(
        graph=graph,
        origin_station=origin,
        destination_station=destination,
        priority=base_priority,
        max_candidates=5,
    )
    active_candidates = search_routes(
        graph=graph,
        origin_station=origin,
        destination_station=destination,
        priority=base_priority,
        max_candidates=5,
        disabled_transfers=disabled_transfers,
        closed_segments=closed_segments,
    )

    if priority == "learned" and trained_model is not None:
        baseline_candidates = rerank_candidates(baseline_candidates, trained_model)
        active_candidates = rerank_candidates(active_candidates, trained_model)

    return baseline_candidates, active_candidates, has_trained_model


def _build_ml_status() -> Dict[str, Any]:
    """PyTorch 好み学習タブの表示に必要な最新ステータス一式を組み立てる."""
    logs = load_choice_logs(app.state.log_path)
    usable_logs = [r for r in logs if len(r.get("candidates", [])) >= 2]
    bundle = load_saved_model(app.state.model_path)
    trained_model = bundle[0] if bundle else None
    trained_summary = bundle[1] if bundle else None

    # 学習済みモデルがある場合、代表 4 区間で「駅数最短順 vs PyTorch 学習後」の 1 位比較を作成
    comparison_rows: List[Dict[str, Any]] = []
    if trained_model is not None:
        compare_pairs = [
            ("浅草", "新宿"),
            ("渋谷", "銀座"),
            ("青山一丁目", "銀座"),
            ("渋谷", "東京"),
        ]
        for o_st, d_st in compare_pairs:
            std_cands = search_routes(graph, o_st, d_st, priority="fewest_stops", max_candidates=4)
            learned_cands = rerank_candidates(std_cands, trained_model)
            if std_cands and learned_cands:
                changed = std_cands[0].signature != learned_cands[0].signature
                comparison_rows.append(
                    {
                        "pair": f"{o_st} → {d_st}",
                        "std_top": f"{std_cands[0].title}（乗換{std_cands[0].transfers}回 / {std_cands[0].stops}区間）",
                        "learned_top": (
                            f"{learned_cands[0].title}（乗換{learned_cands[0].transfers}回 / "
                            f"{learned_cands[0].stops}区間 / スコア{learned_cands[0].score:+.2f}）"
                        ),
                        "changed": changed,
                    }
                )

    # 直近 20 件のログを新しい順でプレビュー用に整形
    recent_logs = [
        {
            "origin": r.get("origin", ""),
            "destination": r.get("destination", ""),
            "chosen_title": r.get("chosen_title", ""),
            "num_candidates": len(r.get("candidates", [])),
            "disabled_transfers": ", ".join(r.get("disabled_transfers", [])) or "-",
        }
        for r in reversed(logs[-20:])
    ]

    return {
        "logs_total": len(logs),
        "usable_logs": len(usable_logs),
        "has_trained_model": trained_model is not None,
        "trained_summary": trained_summary,
        "loss_history": list(app.state.last_loss_history),
        "comparison_rows": comparison_rows,
        "recent_logs": recent_logs,
    }


# ==============================================================================
# HTML エントリーポイント & API ルート定義
# ==============================================================================
@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """トップページ（SPA 風の単一 HTML）を返す."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def get_meta() -> Dict[str, Any]:
    """UI 初期化に必要な駅一覧・乗換一覧・区間一覧・モデル有無を返す."""
    stations = [
        {"name": name, "label": graph.get_station_display_label(name)}
        for name in graph.get_station_names()
    ]
    transfers = [
        {
            "id": tr.id,
            "label": tr.label,
            "note": tr.note,
            "is_diff_name": tr.is_diff_name,
        }
        for tr in graph.get_ordered_transfers()
    ]
    segments = [
        {
            "id": seg.id,
            "label": seg.label,
            "line_id": seg.line_id,
        }
        for seg in graph.get_ordered_segments()
    ]
    has_trained_model = load_saved_model(app.state.model_path) is not None
    return {
        "stations": stations,
        "transfers": transfers,
        "segments": segments,
        "has_trained_model": has_trained_model,
    }


@app.post("/api/search")
def api_search_routes(req: SearchRequest) -> Dict[str, Any]:
    """経路探索を実行し、候補一覧・変更前後比較データ・最新 SVG 路線図をまとめて返す."""
    baseline_candidates, active_candidates, has_trained_model = _run_search_pair(
        origin=req.origin,
        destination=req.destination,
        priority=req.priority,
        disabled_transfers=req.disabled_transfers,
        closed_segments=req.closed_segments,
    )

    selected_idx = req.selected_index if req.selected_index < len(active_candidates) else 0
    selected_route = active_candidates[selected_idx] if active_candidates else None
    baseline_top = baseline_candidates[0] if baseline_candidates else None
    has_constraints = bool(req.disabled_transfers or req.closed_segments)

    svg_html = render_metro_svg(
        graph=graph,
        active_route=selected_route,
        baseline_route=baseline_top if has_constraints else None,
        disabled_transfers=req.disabled_transfers,
        closed_segments=req.closed_segments,
        origin_station=req.origin,
        destination_station=req.destination,
    )

    # 制約によって除外された候補名を抽出
    active_sigs = {c.signature for c in active_candidates}
    dropped_titles = [c.title for c in baseline_candidates if c.signature not in active_sigs]

    # 変更前後の第 1 候補の差分サマリー
    comparison: Optional[Dict[str, Any]] = None
    if has_constraints:
        new_top = active_candidates[0] if active_candidates else None
        status = "unreachable"
        delta_transfers = 0
        delta_stops = 0
        if new_top is not None:
            if baseline_top is not None and new_top.signature == baseline_top.signature:
                status = "unchanged"
            else:
                status = "changed"
                if baseline_top is not None:
                    delta_transfers = new_top.transfers - baseline_top.transfers
                    delta_stops = new_top.stops - baseline_top.stops
        comparison = {
            "baseline_top": _serialize_candidate(baseline_top) if baseline_top else None,
            "active_top": _serialize_candidate(new_top) if new_top else None,
            "status": status,
            "delta_transfers": delta_transfers,
            "delta_stops": delta_stops,
            "dropped_titles": dropped_titles,
        }

    return {
        "origin": req.origin,
        "destination": req.destination,
        "same_station": req.origin == req.destination,
        "priority": req.priority,
        "has_trained_model": has_trained_model,
        "has_constraints": has_constraints,
        "selected_index": selected_idx,
        "candidates": [_serialize_candidate(c) for c in active_candidates],
        "comparison": comparison,
        "svg": svg_html,
    }


@app.post("/api/choices")
def api_record_choice(req: ChoiceLogRequest) -> Dict[str, Any]:
    """ユーザーが選んだ経路を JSONL ログへ追記する."""
    _, active_candidates, _ = _run_search_pair(
        origin=req.origin,
        destination=req.destination,
        priority=req.priority,
        disabled_transfers=req.disabled_transfers,
        closed_segments=req.closed_segments,
    )
    if not active_candidates or req.chosen_index >= len(active_candidates):
        raise HTTPException(status_code=400, detail="選択された候補インデックスが不正です。")

    record = append_choice_log(
        origin=req.origin,
        destination=req.destination,
        priority_mode=req.priority,
        disabled_transfers=req.disabled_transfers,
        closed_segments=req.closed_segments,
        candidates=active_candidates,
        chosen_index=req.chosen_index,
        log_path=app.state.log_path,
    )
    return {
        "message": f"選択を記録しました：{record['chosen_title']}（{req.origin}→{req.destination}）",
        "chosen_title": record["chosen_title"],
        "ml_status": _build_ml_status(),
    }


@app.get("/api/ml/status")
def api_ml_status() -> Dict[str, Any]:
    """選択ログ件数・学習済みモデルの重み・代表ルートの比較結果を返す."""
    return _build_ml_status()


@app.post("/api/ml/train")
def api_ml_train(req: TrainRequest) -> Dict[str, Any]:
    """蓄積された選択ログから PyTorch 好み学習モデルを学習して保存する."""
    logs = load_choice_logs(app.state.log_path)
    try:
        _, summary = train_preference_model(
            records=logs,
            epochs=req.epochs,
            lr=req.lr,
            weight_decay=req.weight_decay,
            save_path=app.state.model_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.state.last_loss_history = summary.loss_history
    return {
        "message": (
            f"学習完了！（Loss: {summary.initial_loss:.4f} → {summary.final_loss:.4f} / "
            f"Top-1一致率: {summary.top1_accuracy * 100:.1f}%）"
        ),
        "ml_status": _build_ml_status(),
    }


@app.post("/api/ml/sample")
def api_ml_sample(req: SampleLogRequest) -> Dict[str, Any]:
    """テスト・体験用の仮想ユーザー選択ログ（乗換回避派 / 駅数最短派）を一括追記する."""
    added = generate_sample_choice_logs(
        graph=graph,
        persona=req.persona,
        log_path=app.state.log_path,
    )
    label = "乗換回避派" if req.persona == "avoid_transfers" else "駅数最短派"
    return {
        "message": f"{label}のサンプル選択を {len(added)} 件追加しました。",
        "added_count": len(added),
        "ml_status": _build_ml_status(),
    }


@app.post("/api/ml/reset")
def api_ml_reset() -> Dict[str, Any]:
    """選択ログと学習済みモデルをすべて削除して初期状態へ戻す."""
    clear_choice_logs(app.state.log_path, app.state.model_path)
    app.state.last_loss_history = []
    return {
        "message": "選択ログと学習済みモデルをリセットしました。",
        "ml_status": _build_ml_status(),
    }


@app.get("/api/ml/logs/download")
def api_download_logs() -> Response:
    """Google Colab 連携用に蓄積済みの user_choices.jsonl をダウンロード配信する."""
    logs = load_choice_logs(app.state.log_path)
    if not logs:
        raise HTTPException(status_code=404, detail="ダウンロードできる選択ログがまだありません。")
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in logs) + "\n"
    return Response(
        content=payload.encode("utf-8"),
        media_type="application/jsonl; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="user_choices.jsonl"'},
    )


if __name__ == "__main__":
    # `python app.py` でそのままローカルサーバーを起動できるエントリーポイント
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
