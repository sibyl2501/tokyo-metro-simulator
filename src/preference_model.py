"""PyTorch preference learning & choice logging for Tokyo Metro routes.

Models user route choices using a Multinomial Logit (listwise cross-entropy)
utility model trained from scratch in PyTorch:

    U(route) = w_stops * stops
             + w_transfers * transfers
             + w_diff * diff_name_transfers
             + w_timing * transfer_timing

From the learned weights, we directly interpret:
    "How many extra stops is the user willing to accept to avoid 1 transfer?"
    = w_transfers / w_stops  (when both weights are negative costs).
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


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_LOG_PATH = DATA_DIR / "user_choices.jsonl"
DEFAULT_MODEL_PATH = DATA_DIR / "preference_model.pt"

FEATURE_NAMES_JA = [
    "乗車区間数 (stops)",
    "乗換回数 (transfers)",
    "異駅名乗換 (diff_name_transfers)",
    "乗換タイミング後半寄り (transfer_timing)",
]
NUM_FEATURES = len(FEATURE_NAMES_JA)


class RoutePreferenceNet(nn.Module):
    """Interpretable linear utility model trained from scratch with PyTorch."""

    def __init__(self, num_features: int = NUM_FEATURES) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=False)
        # Start from neutral zero weights ("初期状態から学習")
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    @property
    def weights(self) -> List[float]:
        return self.linear.weight.detach().cpu().view(-1).tolist()


@dataclass
class TrainingSummary:
    num_events_total: int
    num_events_used: int
    epochs: int
    initial_loss: float
    final_loss: float
    top1_accuracy: float
    loss_history: List[float]
    weights: Dict[str, float]
    transfer_equivalent_stops: Optional[float]
    interpretation: str


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
    """Append a user route choice event to JSONL."""
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
    if log_path.exists():
        log_path.unlink()
    if model_path.exists():
        model_path.unlink()


def _interpret_weights(w: List[float]) -> Tuple[Optional[float], str]:
    w_stops, w_transfers, w_diff, w_timing = w
    eq_stops: Optional[float] = None
    lines: List[str] = []

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
    """Train a PyTorch listwise choice model from scratch on logged user choices."""
    usable = [
        r
        for r in records
        if isinstance(r.get("candidates"), list)
        and len(r["candidates"]) >= 2
        and 0 <= int(r.get("chosen_index", -1)) < len(r["candidates"])
    ]
    if not usable:
        raise ValueError("学習には2候補以上ある選択ログが最低1件必要です。")

    torch.manual_seed(42)
    model = RoutePreferenceNet(num_features=NUM_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tensors = [
        (
            torch.tensor([c["features"] for c in r["candidates"]], dtype=torch.float32),
            torch.tensor(int(r["chosen_index"]), dtype=torch.long),
        )
        for r in usable
    ]

    loss_history: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for feat_mat, target_idx in tensors:
            logits = model(feat_mat)
            total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target_idx.unsqueeze(0))
        mean_loss = total_loss / len(tensors)
        mean_loss.backward()
        optimizer.step()
        loss_history.append(float(mean_loss.item()))

    # Evaluate Top-1 choice accuracy
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
    if not model_path.exists():
        return None
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = RoutePreferenceNet(num_features=NUM_FEATURES)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("summary", {})


def rerank_candidates(
    candidates: Sequence[RouteCandidate],
    model: RoutePreferenceNet,
) -> List[RouteCandidate]:
    """Score and re-rank route candidates using the trained PyTorch model."""
    if not candidates:
        return []
    model.eval()
    with torch.no_grad():
        feat_mat = torch.tensor([c.feature_vector for c in candidates], dtype=torch.float32)
        scores = model(feat_mat).cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]

    scored = list(candidates)
    for cand, sc in zip(scored, scores):
        cand.score = round(float(sc), 4)
    scored.sort(key=lambda c: (-(c.score if c.score is not None else 0.0), c.transfers, c.stops))
    return scored


def generate_sample_choice_logs(
    graph: MetroGraph,
    persona: str = "avoid_transfers",
    log_path: Optional[Path] = DEFAULT_LOG_PATH,
) -> List[dict]:
    """Generate realistic synthetic choice records for quick testing / Colab demos.

    Personas:
      - 'avoid_transfers': dislikes transfers (~3.5 stops penalty) & diff-name transfers
      - 'shortest_stops': strongly prefers fewest stops (~0.4 stops penalty for transfer)
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
        # Strongly prioritizes fewest stops; tolerates 1 transfer even to save 1 stop
        true_w = (-1.5, -0.5, -0.3, -0.2)
    else:
        # Avoids transfers (willing to ride ~3.5 extra stops to avoid 1 transfer)
        # and dislikes walking through different-name transfer passages
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
