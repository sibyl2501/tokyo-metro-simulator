"""PyTorch を用いた経路好み学習 & 選択ログ管理モジュール.

ユーザーがシミュレーター上で「👍 私ならこの経路を選ぶ」を押して蓄積した選択ログから、
多項ロジットモデル（Multinomial Logit / Listwise Cross-Entropy）を PyTorch でゼロから学習します。

【モデルの効用関数】
各候補経路の「好みスコア（効用）」U(route) を 4 つの特徴量の線形結合で表現します。

    U(route) = w_stops     * stops                （乗車区間数）
             + w_transfers * transfers            （乗換回数）
             + w_diff      * diff_name_transfers  （異駅名乗換の回数）
             + w_timing    * transfer_timing      （乗換タイミングが後半寄りか）

【重みの解釈（なぜ線形モデルを使うのか）】
w_stops と w_transfers がどちらも負（少ないほど好ましいコスト）として学習されたとき、
その比率 `w_transfers / w_stops` を取るだけで、
  「乗換を 1 回減らすためなら、何駅分の遠回りまで許容するか」
を人間が直感的に理解できる数値としてそのまま読み取ることができます。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metro_graph import MetroGraph
from src.routing import RouteCandidate, search_routes


# ログファイル（JSONL）および学習済みモデル重み（.pt）のデフォルト保存先
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_LOG_PATH = DATA_DIR / "user_choices.jsonl"
DEFAULT_MODEL_PATH = DATA_DIR / "preference_model.pt"

# 特徴量ベクトルの各次元に対応する日本語ラベル（グラフ表示・モデル保存用）
FEATURE_NAMES_JA = [
    "乗車区間数 (stops)",
    "乗換回数 (transfers)",
    "異駅名乗換 (diff_name_transfers)",
    "乗換タイミング後半寄り (transfer_timing)",
]
NUM_FEATURES = len(FEATURE_NAMES_JA)


class RoutePreferenceNet(nn.Module):
    """解釈しやすい線形効用モデル（PyTorch nn.Module）.

    多項ロジット（Softmax）では各候補のスコア差だけが選択確率に効くため、
    定数項（bias）は不要（bias=False）としています。
    また初期重みをすべて 0.0 で初期化することで、先入観なしの状態から学習を始めます。
    """

    def __init__(self, num_features: int = NUM_FEATURES) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=False)
        # 初期状態から学習するため、重みを中立の 0 で初期化
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴量行列 [候補数, 特徴量数] を受け取り、各候補の効用スコア [候補数] を返す."""
        return self.linear(x).squeeze(-1)

    @property
    def weights(self) -> List[float]:
        """学習された各特徴量の重みを Python の float リストとして取り出す."""
        return self.linear.weight.detach().cpu().view(-1).tolist()


@dataclass
class TrainingSummary:
    """1 回の学習ジョブの結果指標と自然言語による解釈をまとめたデータクラス."""

    num_events_total: int                     # 読み込んだ選択ログ総数
    num_events_used: int                      # 実際に学習に使えた件数（2候補以上あるもの）
    epochs: int                               # 学習エポック数
    initial_loss: float                       # 学習開始時（エポック 1）の平均 Loss
    final_loss: float                         # 学習終了時の平均 Loss
    top1_accuracy: float                      # 学習データ上で 1 位予測が一致した割合
    loss_history: List[float]                 # エポックごとの Loss 推移（グラフ描画用）
    weights: Dict[str, float]                 # 特徴量名 -> 学習済み重み
    transfer_equivalent_stops: Optional[float]  # 「乗換 1 回 = 何駅分相当か」の推定値
    interpretation: str                       # 重みから生成した日本語の傾向解説文


def append_choice_log(
    origin: str,
    destination: str,
    priority_mode: str,
    disabled_transfers: Sequence[str],
    closed_segments: Sequence[str],
    candidates: Sequence[RouteCandidate],
    chosen_index: int,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    """ユーザーが選んだ 1 回分の経路選択イベントを JSONL ファイルに追記保存する.

    JSONL（1 行 1 JSON）形式にすることで、1 クリックごとに末尾追記でき、
    Google Colab のノートブックにもそのままアップロードして読み込めます。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "origin": origin,
        "destination": destination,
        "priority_mode": priority_mode,
        "disabled_transfers": list(disabled_transfers),
        "closed_segments": list(closed_segments),
        "chosen_index": int(chosen_index),
        "chosen_title": candidates[chosen_index].title,
        "candidates": [c.to_log_dict() for c in candidates],
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_choice_logs(log_path: Path = DEFAULT_LOG_PATH) -> List[dict]:
    """JSONL ファイルから記録済みの選択ログ一覧を読み込む（壊れた行は無視する）."""
    if not log_path.exists():
        return []
    records: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def clear_choice_logs(
    log_path: Path = DEFAULT_LOG_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """選択ログファイルと学習済みモデルファイルを削除して学習状態をリセットする."""
    if log_path.exists():
        log_path.unlink()
    if model_path.exists():
        model_path.unlink()


def _interpret_weights(w: List[float]) -> Tuple[Optional[float], str]:
    """学習された 4 つの重みベクトルを、ユーザー向けのわかりやすい日本語解説に変換する."""
    w_stops, w_transfers, w_diff, w_timing = w
    eq_stops: Optional[float] = None
    lines: List[str] = []

    # 乗車区間数と乗換回数の重みがどちらも負であれば、その比率から許容遠回り駅数を算出
    if w_stops < -1e-4 and w_transfers < -1e-4:
        eq_stops = w_transfers / w_stops
        lines.append(
            f"乗換を1回減らすためなら、約 {eq_stops:.1f} 駅分までの遠回りを許容する傾向があります。"
        )
    elif w_transfers < -1e-4 and w_stops >= -1e-4:
        lines.append("乗車区間数よりも「乗換回数を減らすこと」を強く重視する傾向があります。")
    elif w_stops < -1e-4 and w_transfers >= -1e-4:
        lines.append("乗換回数よりも「乗車区間数を短くすること」を強く重視する傾向があります。")
    else:
        lines.append("まだ十分なトレードオフ事例が少ないため、特徴の傾向は中立的です。")

    # 異駅名乗換（連絡通路が長い乗換）に対する好み・回避傾向
    if w_diff < -0.15:
        if w_stops < -1e-4:
            diff_eq = w_diff / w_stops
            lines.append(
                f"異駅名乗換（溜池山王↔国会議事堂前など）を避ける傾向があります（約 {diff_eq:.1f} 駅分相当）。"
            )
        else:
            lines.append("異駅名乗換（連絡通路が長い乗換）を避ける傾向があります。")
    elif w_diff > 0.15:
        lines.append("異駅名乗換をあまり気にせず選ぶ傾向があります。")

    # 乗換タイミング（序盤で乗り換えたいか／後半まで乗り続けたいか）
    if w_timing < -0.15:
        lines.append("乗換がある場合は、なるべく前半（早い段階）で乗り換える経路を好む傾向です。")
    elif w_timing > 0.15:
        lines.append("乗換がある場合は、なるべく後半まで乗ってから乗り換える経路を好む傾向です。")

    return eq_stops, " ".join(lines)


def train_preference_model(
    records: Sequence[dict],
    epochs: int = 250,
    lr: float = 0.08,
    weight_decay: float = 0.02,
    save_path: Optional[Path] = DEFAULT_MODEL_PATH,
) -> Tuple[RoutePreferenceNet, TrainingSummary]:
    """蓄積された選択ログから PyTorch モデルを学習し、モデル本体と学習サマリーを返す.

    【学習の仕組み（Listwise Cross-Entropy）】
    1 回の検索イベントで表示された M 個の候補をそれぞれ特徴量行列 X (M × 4) に変換し、
    モデルが出力する M 個のロジット（効用スコア）に対して
    「実際に選ばれた候補インデックス chosen_index を正解クラスとする Cross-Entropy 損失」
    を最小化します。weight_decay（L2 正則化）を入れることで少量ログでも重みが発散しないようにしています。
    """
    # 候補が 2 件以上あり、chosen_index が有効範囲内のレコードだけを学習対象とする
    usable = [
        r
        for r in records
        if isinstance(r.get("candidates"), list)
        and len(r["candidates"]) >= 2
        and 0 <= int(r.get("chosen_index", -1)) < len(r["candidates"])
    ]
    if not usable:
        raise ValueError("学習には2候補以上ある選択ログが最低1件必要です。")

    # 再現性を保つため乱数シードを固定
    torch.manual_seed(42)
    model = RoutePreferenceNet(num_features=NUM_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 各選択イベントを事前に PyTorch テンソル化しておく（エポック内の再変換を避けて高速化）
    tensors = [
        (
            torch.tensor([c["features"] for c in r["candidates"]], dtype=torch.float32),
            torch.tensor(int(r["chosen_index"]), dtype=torch.long),
        )
        for r in usable
    ]

    # 全バッチ勾配降下法（ログ数が数十〜数百件規模なので 1 エポックで全件の平均損失を更新）
    loss_history: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for feat_mat, target_idx in tensors:
            logits = model(feat_mat)
            # F.cross_entropy は [バッチサイズ=1, 候補数] と正解クラス [1] を受け取る
            total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target_idx.unsqueeze(0))
        mean_loss = total_loss / len(tensors)
        mean_loss.backward()
        optimizer.step()
        loss_history.append(float(mean_loss.item()))

    # 学習後のモデルが各選択イベントで選んだ 1 位と実際の選択がどれだけ一致するか（Top-1 一致率）を評価
    correct = 0
    with torch.no_grad():
        for feat_mat, target_idx in tensors:
            pred = int(torch.argmax(model(feat_mat)).item())
            if pred == int(target_idx.item()):
                correct += 1
    acc = correct / len(tensors)

    w_list = model.weights
    eq_stops, interpretation = _interpret_weights(w_list)
    weights_dict = {name: round(val, 4) for name, val in zip(FEATURE_NAMES_JA, w_list)}

    summary = TrainingSummary(
        num_events_total=len(records),
        num_events_used=len(usable),
        epochs=epochs,
        initial_loss=round(loss_history[0], 4),
        final_loss=round(loss_history[-1], 4),
        top1_accuracy=round(acc, 4),
        loss_history=loss_history,
        weights=weights_dict,
        transfer_equivalent_stops=round(eq_stops, 2) if eq_stops is not None else None,
        interpretation=interpretation,
    )

    # save_path が指定されていれば、学習済み重みとサマリー辞書を .pt ファイルに保存
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES_JA,
                "summary": {
                    "num_events_total": summary.num_events_total,
                    "num_events_used": summary.num_events_used,
                    "epochs": summary.epochs,
                    "initial_loss": summary.initial_loss,
                    "final_loss": summary.final_loss,
                    "top1_accuracy": summary.top1_accuracy,
                    "weights": summary.weights,
                    "transfer_equivalent_stops": summary.transfer_equivalent_stops,
                    "interpretation": summary.interpretation,
                },
            },
            save_path,
        )

    return model, summary


def load_saved_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Optional[Tuple[RoutePreferenceNet, dict]]:
    """保存済みの .pt チェックポイントからモデルとサマリーを読み込む（未学習時は None）."""
    if not model_path.exists():
        return None
    # 安全のため weights_only=True でロードし、推論モード (eval) に切り替える
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = RoutePreferenceNet(num_features=NUM_FEATURES)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("summary", {})


def rerank_candidates(
    candidates: Sequence[RouteCandidate],
    model: RoutePreferenceNet,
) -> List[RouteCandidate]:
    """学習済み PyTorch モデルで各経路候補の好みスコアを計算し、スコア降順に並べ替える."""
    if not candidates:
        return []
    model.eval()
    with torch.no_grad():
        feat_mat = torch.tensor([c.feature_vector for c in candidates], dtype=torch.float32)
        scores = model(feat_mat).cpu().tolist()
    # 候補が 1 件だけのとき squeeze(-1) でスカラー float が返るためリストに揃える
    if isinstance(scores, float):
        scores = [scores]

    scored = list(candidates)
    for cand, sc in zip(scored, scores):
        cand.score = round(float(sc), 4)
    # スコアが高い順（同点時は乗換が少ない順→駅数が少ない順）に並べ替え
    scored.sort(key=lambda c: (-(c.score if c.score is not None else 0.0), c.transfers, c.stops))
    return scored


def generate_sample_choice_logs(
    graph: MetroGraph,
    persona: str = "avoid_transfers",
    log_path: Optional[Path] = DEFAULT_LOG_PATH,
) -> List[dict]:
    """アプリ体験・自動テスト・Colab デモ用に、仮想ユーザーの選択ログを一括生成する.

    2 種類のペルソナ（真の好み重み）を用意し、代表的な 15 区間の検索結果から最も効用の高い候補を選ばせます。
      - 'avoid_transfers': 乗換回避派（乗換 1 回を避けるため約 3.5 駅分の遠回りを許容、異駅名乗換も嫌う）
      - 'shortest_stops' : 駅数最短派（1 駅でも短くなるなら積極的に乗り換える）
    """
    sample_queries: List[Tuple[str, str, List[str], List[str]]] = [
        ("浅草", "新宿", [], []),
        ("渋谷", "銀座", [], []),
        ("青山一丁目", "銀座", [], []),
        ("表参道", "銀座", [], []),
        ("溜池山王", "銀座", [], []),
        ("渋谷", "東京", [], []),
        ("浅草", "池袋", [], []),
        ("上野", "新宿", [], []),
        ("日本橋", "四ツ谷", [], []),
        ("外苑前", "霞ケ関", [], []),
        ("新橋", "赤坂見附", [], []),
        ("方南町", "新橋", [], []),
        ("新宿", "東京", [], ["M-14__M-15"]),
        ("浅草", "新宿", ["transfer:銀座"], []),
        ("渋谷", "大手町", [], []),
    ]

    if persona == "shortest_stops":
        # 乗車区間数へのペナルティが乗換ペナルティの 3 倍大きく、1 駅短縮のために乗換を厭わない
        true_w = (-1.5, -0.5, -0.3, -0.2)
    else:
        # 乗換ペナルティが大きく（-2.2 / -0.6 ≈ 3.7 駅分）、異駅名乗換の連絡通路も避ける
        true_w = (-0.6, -2.2, -1.0, -0.4)

    created: List[dict] = []
    for origin, dest, dis_tr, cls_seg in sample_queries:
        cands = search_routes(
            graph=graph,
            origin_station=origin,
            destination_station=dest,
            priority="fewest_transfers",
            max_candidates=5,
            disabled_transfers=dis_tr,
            closed_segments=cls_seg,
        )
        if len(cands) < 2:
            continue
        # ペルソナの真の重みと各候補の特徴量の内積を計算し、最大スコアの候補を選択
        scores = [
            sum(w * f for w, f in zip(true_w, c.feature_vector))
            for c in cands
        ]
        chosen_idx = max(range(len(cands)), key=lambda i: scores[i])
        if log_path is not None:
            rec = append_choice_log(
                origin=origin,
                destination=dest,
                priority_mode="fewest_transfers",
                disabled_transfers=dis_tr,
                closed_segments=cls_seg,
                candidates=cands,
                chosen_index=chosen_idx,
                log_path=log_path,
            )
            created.append(rec)
        else:
            # テスト時などファイルに書き出したくない場合はメモリ上の辞書リストだけ返す
            created.append(
                {
                    "origin": origin,
                    "destination": dest,
                    "chosen_index": chosen_idx,
                    "chosen_title": cands[chosen_idx].title,
                    "candidates": [c.to_log_dict() for c in cands],
                }
            )
    return created
