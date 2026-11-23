"""FastAPI エンドポイントの統合テスト集.

TestClient を使って、トップページ配信・メタ情報取得・経路検索・変更前後比較・
選択ログ記録・PyTorch 学習・JSONL ダウンロード・リセットの一連の API 動作を検証します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import app as web_app


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    """テストごとに独立した一時ファイルパスへ切り替えた TestClient を提供する."""
    orig_log = web_app.app.state.log_path
    orig_model = web_app.app.state.model_path
    orig_hist = web_app.app.state.last_loss_history

    web_app.app.state.log_path = tmp_path / "user_choices.jsonl"
    web_app.app.state.model_path = tmp_path / "preference_model.pt"
    web_app.app.state.last_loss_history = []
    try:
        with TestClient(web_app.app) as c:
            yield c
    finally:
        web_app.app.state.log_path = orig_log
        web_app.app.state.model_path = orig_model
        web_app.app.state.last_loss_history = orig_hist


def test_index_and_meta(client: TestClient) -> None:
    """トップページ HTML と /api/meta の基本構成を検証する."""
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert "東京メトロ 乗換シミュレーター" in res_home.text

    res_meta = client.get("/api/meta")
    assert res_meta.status_code == 200
    meta = res_meta.json()
    assert len(meta["stations"]) == 45
    assert len(meta["transfers"]) == 3
    assert len(meta["segments"]) > 0
    assert meta["has_trained_model"] is False


def test_search_and_disruption_comparison(client: TestClient) -> None:
    """通常検索・銀座乗換無効化時の変更前後比較・不正駅エラーを検証する."""
    # 通常検索：浅草 -> 新宿
    res = client.post(
        "/api/search",
        json={"origin": "浅草", "destination": "新宿", "priority": "fewest_transfers"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["candidates"][0]["title"] == "銀座経由"
    assert data["comparison"] is None
    assert data["svg"].startswith("<svg")

    # 銀座乗換NG：代替経路へ切り替わり、comparison に差分と除外候補が入る
    res_ng = client.post(
        "/api/search",
        json={
            "origin": "浅草",
            "destination": "新宿",
            "priority": "fewest_transfers",
            "disabled_transfers": ["transfer:銀座"],
        },
    )
    assert res_ng.status_code == 200
    ng = res_ng.json()
    assert ng["candidates"][0]["title"] != "銀座経由"
    assert ng["comparison"]["status"] == "changed"
    assert "銀座経由" in ng["comparison"]["dropped_titles"]

    # 存在しない駅名は 400 Bad Request
    res_err = client.post(
        "/api/search",
        json={"origin": "存在しない駅", "destination": "新宿"},
    )
    assert res_err.status_code == 400


def test_ml_lifecycle_end_to_end(client: TestClient) -> None:
    """選択ログ記録 -> サンプル投入 -> PyTorch 学習 -> 再ランキング検索 -> ダウンロード -> リセットの一連の流れを検証する."""
    # 1. 手動選択ログの記録
    rec_res = client.post(
        "/api/choices",
        json={
            "origin": "渋谷",
            "destination": "銀座",
            "priority": "fewest_transfers",
            "chosen_index": 0,
        },
    )
    assert rec_res.status_code == 200
    assert rec_res.json()["ml_status"]["logs_total"] == 1

    # 2. 乗換回避派サンプル投入
    sample_res = client.post("/api/ml/sample", json={"persona": "avoid_transfers"})
    assert sample_res.status_code == 200
    assert sample_res.json()["added_count"] > 5

    # 3. PyTorch モデル学習
    train_res = client.post("/api/ml/train", json={"epochs": 120})
    assert train_res.status_code == 200
    ml_status = train_res.json()["ml_status"]
    assert ml_status["has_trained_model"] is True
    assert len(ml_status["loss_history"]) == 120
    assert len(ml_status["comparison_rows"]) == 4

    # 4. 学習済みモデルでの再ランキング検索
    learned_res = client.post(
        "/api/search",
        json={"origin": "渋谷", "destination": "銀座", "priority": "learned"},
    )
    assert learned_res.status_code == 200
    learned_data = learned_res.json()
    assert learned_data["has_trained_model"] is True
    assert learned_data["candidates"][0]["score"] is not None
    assert learned_data["candidates"][0]["transfers"] == 0

    # 5. JSONL ダウンロード
    dl_res = client.get("/api/ml/logs/download")
    assert dl_res.status_code == 200
    assert "user_choices.jsonl" in dl_res.headers["content-disposition"]
    assert "渋谷" in dl_res.text

    # 6. リセット
    reset_res = client.post("/api/ml/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["ml_status"]["logs_total"] == 0
    assert reset_res.json()["ml_status"]["has_trained_model"] is False
