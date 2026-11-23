"""東京メトロ路線網のグラフ表現モジュール.

このモジュールでは、路線網を「駅名」単位ではなく「駅 × 路線」単位のノードとして表現します。
例:
  - 銀座（銀座線）   -> ノード ID: "G-09"
  - 銀座（丸ノ内線） -> ノード ID: "M-16"

このようにノードを分離し、エッジ（辺）を次の 2 種類に分けることがこのアプリの核心です。
  1. 乗車エッジ ("ride")    : 同じ路線上の隣駅同士を結ぶ（列車に乗る移動）
  2. 乗換エッジ ("transfer"): 異なる路線のホーム同士を結ぶ（徒歩での乗り換え）

この設計により、以下の要件をシンプルかつ正確に実現できます。
  - 「銀座駅での乗換だけ無効にし、銀座駅を通過する列車は生かす」
    -> 銀座の transfer エッジだけを通行不可にすれば実現できる
  - 「溜池山王 ↔ 国会議事堂前」のような異駅名での公式乗換
    -> 駅名が違っていても transfer エッジを張るだけで自然に表現できる
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# デフォルトの路線データ JSON ファイルのパス（プロジェクトルート/data/tokyo_metro_mvp.json）
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tokyo_metro_mvp.json"


@dataclass(frozen=True)
class LineInfo:
    """路線の基本情報（路線名・公式カラー・方面表示名など）を保持する不変データクラス."""

    id: str                         # 路線記号（例: "G", "M"）
    name: str                       # 日本語の路線名（例: "銀座線"）
    name_en: str                    # 英語の路線名
    color: str                      # 路線の公式シンボルカラー（HEX）
    text_color: str                 # バッジ等に乗せる文字色
    forward_direction: str          # 駅番号が増える方向の方面名
    backward_direction: str         # 駅番号が減る方向の方面名
    branch_direction: Optional[str] = None  # 支線がある場合の方面名（例: 丸ノ内線の方南町方面）


@dataclass(frozen=True)
class StationNode:
    """「駅 × 路線」を表すグラフの頂点（ノード）データクラス.

    SVG 路線図を描画するための座標 (x, y) や、駅名ラベルのオフセット (lx, ly) も併せて保持します。
    """

    id: str                 # 一意のノード ID（例: "G-09", "M-16", "Mb-03"）
    code: str               # 駅ナンバリング表記（例: "G-09", "Mb-03"）
    station_name: str       # 日本語の駅名（例: "銀座"）
    station_name_en: str    # 英語の駅名
    line_id: str            # 所属する路線 ID（例: "G", "M"）
    branch: str             # 本線 ("main") か支線 ("honancho" 等) かの区分
    x: float                # SVG 上の駅ノード X 座標
    y: float                # SVG 上の駅ノード Y 座標
    lx: float               # 駅名ラベルの X 方向オフセット
    ly: float               # 駅名ラベルの Y 方向オフセット
    anchor: str             # SVG text-anchor ("start" / "middle" / "end")
    is_major: bool = False  # 主要駅フラグ（丸を大きく・太字で描画する）
    hide_label: bool = False  # 同一駅名で 2 路線が重なる場合など、片方のラベル描画を省略するフラグ


@dataclass(frozen=True)
class Edge:
    """ノード間を結ぶ有向エッジ（乗車または乗換の 1 ステップ）."""

    source: str                           # 出発ノード ID
    target: str                           # 到着ノード ID
    kind: str                             # "ride"（乗車）または "transfer"（乗換）
    line_id: Optional[str] = None         # 乗車エッジの場合の路線 ID
    direction_label: Optional[str] = None # 乗車エッジの場合の方面名（例: "渋谷方面"）
    segment_id: Optional[str] = None      # 乗車エッジに対応する無向区間 ID（区間閉鎖判定用）
    transfer_id: Optional[str] = None     # 乗換エッジに対応する公式乗換 ID（乗換無効化判定用）
    transfer_label: Optional[str] = None  # 乗換箇所の表示名（例: "溜池山王 ↔ 国会議事堂前"）
    is_diff_name: bool = False            # 異駅名間の連絡通路乗換なら True


@dataclass(frozen=True)
class TransferInfo:
    """公式乗換ポイント（UI のチェックボックスや SVG 描画の単位）."""

    id: str            # 一意の乗換 ID（例: "transfer:銀座", "transfer:溜池山王-国会議事堂前"）
    node_a: str        # 接続する片方のノード ID
    node_b: str        # 接続するもう片方のノード ID
    label: str         # UI 表示用ラベル
    is_diff_name: bool # 異駅名乗換かどうか
    note: str = ""     # 補足説明（ツールチップ等に表示）


@dataclass(frozen=True)
class SegmentInfo:
    """隣り合う 2 駅間の線路区間（区間閉鎖モードで双方向まとめて閉じる単位）."""

    id: str       # 正規化された無向区間 ID（例: "M-14__M-15"）
    node_a: str   # 端点のノード ID A
    node_b: str   # 端点のノード ID B
    line_id: str  # 路線 ID
    label: str    # UI 表示用ラベル（例: "丸ノ内線：国会議事堂前 — 霞ケ関"）


def make_segment_id(node_a: str, node_b: str) -> str:
    """2 つのノード ID から、向きに依存しない正規化済みの無向区間 ID を生成する.

    A->B でも B->A でも同じ ID（例: "M-14__M-15"）になるように辞書順ソートして結合します。
    これにより「区間閉鎖」で 1 つの ID を指定するだけで上下線まとめて閉鎖できます。
    """
    a, b = sorted((node_a, node_b))
    return f"{a}__{b}"


class MetroGraph:
    """東京メトロの「駅 × 路線」ネットワークをメモリ上で管理するグラフクラス."""

    def __init__(self, raw_data: dict) -> None:
        """JSON から読み込んだ生辞書データをもとに、ノード・隣接リスト・区間・乗換情報を構築する."""
        self.metadata: dict = raw_data.get("metadata", {})

        # 1. 路線マスターの読み込み（路線 ID -> LineInfo）
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

        # 2. 駅×路線ノードの登録と、「駅名 -> 所属ノード一覧」の逆引き辞書作成
        #    例: "銀座" -> ["G-09", "M-16"]（出発駅・目的駅の候補ノード展開に使用）
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

        # 隣接リスト（ダイクストラ法が各ノードから辿れるエッジ一覧を高速に引くための構造）
        self.adjacency: Dict[str, List[Edge]] = {node_id: [] for node_id in self.nodes}
        self.segments: Dict[str, SegmentInfo] = {}
        self.transfers: Dict[str, TransferInfo] = {}

        # 3. 乗車エッジ（隣駅間）の双方向登録
        #    1 つの区間定義に対して「順方向 (u -> v)」と「逆方向 (v -> u)」の 2 本の有向エッジを張る。
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
            # 順方向（例: 浅草 -> 田原町・渋谷方面）
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
            # 逆方向（例: 田原町 -> 浅草・浅草方面）
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

        # 4. 公式乗換エッジの双方向登録（A -> B と B -> A の両方向へ乗り換え可能）
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
        """JSON ファイルを読み込んで MetroGraph インスタンスを生成するファクトリメソッド."""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return cls(raw_data)

    def get_station_names(self) -> List[str]:
        """重複を除いた駅名一覧を JSON の定義順（路線順）で返す."""
        seen: List[str] = []
        for node in self.nodes.values():
            if node.station_name not in seen:
                seen.append(node.station_name)
        return seen

    def get_station_display_label(self, station_name: str) -> str:
        """ドロップダウン表示用に駅ナンバリング付きの文字列を返す（例: '銀座 (G-09/M-16)'）."""
        node_ids = self.station_name_to_nodes[station_name]
        codes = "/".join(self.nodes[nid].code for nid in node_ids)
        return f"{station_name} ({codes})"

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """指定した 2 ノード間を直接結ぶエッジを取得する（存在しない場合は None）."""
        for edge in self.adjacency.get(source, []):
            if edge.target == target:
                return edge
        return None

    def get_ordered_segments(self) -> List[SegmentInfo]:
        """全乗車区間を定義順のリストで返す（区間閉鎖マルチセレクトの選択肢用）."""
        return list(self.segments.values())

    def get_ordered_transfers(self) -> List[TransferInfo]:
        """全公式乗換ポイントを定義順のリストで返す（乗換無効化チェックボックス用）."""
        return list(self.transfers.values())

    def get_line_colors_for_station(self, station_name: str) -> List[Tuple[str, str]]:
        """指定した駅に乗り入れている (路線 ID, 路線カラー) のリストを返す."""
        return [
            (self.nodes[nid].line_id, self.lines[self.nodes[nid].line_id].color)
            for nid in self.station_name_to_nodes.get(station_name, [])
        ]
