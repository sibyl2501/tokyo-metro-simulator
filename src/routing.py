"""東京メトロ経路探索エンジン（ダイクストラ法 ＋ Yen の K 最短単純経路探索）.

このモジュールは、MetroGraph 上で現実的かつ多様な経路候補を探索します。

主な機能:
  1. 辞書順コストによる 2 つの優先モード
     - 'fewest_transfers': (乗換回数, 乗車区間数) の順で最小化
     - 'fewest_stops'    : (乗車区間数, 乗換回数) の順で最小化
  2. 制約シミュレーション
     - 乗換無効化（disabled_transfers）: 指定駅の乗換エッジのみ除外（通過列車は利用可）
     - 区間閉鎖（closed_segments）      : 指定した隣駅間（上下線）の乗車エッジを除外
  3. 多様候補の生成（Yen のアルゴリズム）
     - 1 位の経路だけでなく、乗換駅が異なる代替候補やトレードオフ候補を複数見つけ出す
     - 同じ駅へ戻ってくる不自然なループ経路や連続乗換を除外する
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.metro_graph import Edge, MetroGraph


@dataclass
class RouteLeg:
    """1 回の乗車区間（乗り換えずに同じ路線・方面に乗り続けるまとまり）."""

    line_id: str             # 路線 ID（例: "G", "M"）
    line_name: str           # 路線名（例: "銀座線"）
    direction_label: str     # 方面名（例: "渋谷方面", "池袋方面"）
    node_ids: List[str]      # 通過するノード ID 列（乗車駅〜降車駅）
    station_names: List[str] # 通過する駅名列（乗車駅〜降車駅）
    stops: int               # この区間での乗車駅数（= len(node_ids) - 1）
    is_branch_leg: bool = False  # 丸ノ内線の方南町支線を含む乗車区間かどうか


@dataclass
class TransferStep:
    """経路中に発生する 1 回の乗換ステップの詳細情報."""

    transfer_id: str       # 公式乗換 ID（例: "transfer:銀座"）
    transfer_label: str    # 表示ラベル（例: "銀座", "溜池山王 ↔ 国会議事堂前"）
    from_node_id: str      # 乗換元のノード ID
    to_node_id: str        # 乗換先のノード ID
    from_station: str      # 乗換元の駅名
    to_station: str        # 乗換先の駅名
    from_line_id: str      # 乗換元の路線 ID
    to_line_id: str        # 乗換先の路線 ID
    is_diff_name: bool     # 異駅名間の連絡通路乗換かどうか
    stop_index_before: int # この乗換が起きるまでに乗車した駅数（乗換タイミング特徴量の計算に使用）


@dataclass
class RouteCandidate:
    """探索結果の 1 つの経路候補（UI 表示・SVG 描画・PyTorch 学習の共通データ構造）."""

    origin_station: str          # 出発駅名
    destination_station: str     # 目的駅名
    node_ids: List[str]          # 辿った全ノード ID 列
    edges: List[Edge]            # 辿った全エッジ列（ride / transfer 混在）
    stops: int                   # 合計乗車区間数（ride エッジの本数）
    transfers: int               # 合計乗換回数（transfer エッジの本数）
    diff_name_transfers: int     # うち異駅名乗換の回数
    transfer_timing: float       # 乗換タイミング指標（-0.5: 序盤乗換 〜 +0.5: 終盤乗換、直通は 0.0）
    legs: List[RouteLeg] = field(default_factory=list)
    transfer_steps: List[TransferStep] = field(default_factory=list)
    title: str = ""              # 候補の短い見出し（例: "銀座経由", "銀座線 直通"）
    guidance: str = ""           # 日本語の乗換案内文
    score: Optional[float] = None  # PyTorch 好み学習モデルで再ランキングした際の効用スコア

    @property
    def signature(self) -> str:
        """経路の一意性を判定するための識別文字列（重複排除や変更前後比較に使用）."""
        return "->".join(self.node_ids)

    @property
    def transfer_ids(self) -> List[str]:
        """この経路で使用している乗換 ID のリスト."""
        return [step.transfer_id for step in self.transfer_steps]

    @property
    def segment_ids(self) -> List[str]:
        """この経路で使用している乗車区間 ID のリスト."""
        return [e.segment_id for e in self.edges if e.kind == "ride" and e.segment_id]

    @property
    def feature_vector(self) -> List[float]:
        """PyTorch の好み学習モデルに入力する 4 次元の特徴量ベクトルを返す.

        特徴量の意味（いずれも人間が解釈しやすい値）:
          0. stops               : 合計乗車区間数
          1. transfers           : 合計乗換回数
          2. diff_name_transfers : 異駅名乗換の回数（連絡通路が長い乗換のペナルティ判定用）
          3. transfer_timing     : 乗換が全行程の前半か後半か（平均位置割合 - 0.5、直通時は 0.0）
        """
        return [
            float(self.stops),
            float(self.transfers),
            float(self.diff_name_transfers),
            float(self.transfer_timing),
        ]

    def to_log_dict(self) -> dict:
        """選択ログ（JSONL）へ保存するための辞書形式に変換する."""
        return {
            "signature": self.signature,
            "title": self.title,
            "guidance": self.guidance,
            "stops": self.stops,
            "transfers": self.transfers,
            "diff_name_transfers": self.diff_name_transfers,
            "transfer_timing": round(self.transfer_timing, 4),
            "transfer_ids": self.transfer_ids,
            "features": self.feature_vector,
        }


def _cost_tuple(stops: int, transfers: int, priority: str) -> Tuple[int, int]:
    """優先モードに応じて、ダイクストラ法で比較する辞書順コストのタプルを返す.

    Python のタプル比較は先頭要素を最優先し、同値のときだけ 2 番目で比較するため、
    「まず乗換回数を最小化し、同じなら駅数が少ない方を選ぶ」といった基準をそのまま表現できます。
    """
    if priority == "fewest_stops":
        return (stops, transfers)
    return (transfers, stops)


def _edge_allowed(
    edge: Edge,
    disabled_transfers: Set[str],
    closed_segments: Set[str],
    blocked_edges: Set[Tuple[str, str]],
) -> bool:
    """指定されたエッジが現在の制約下で通行可能かどうかを判定する."""
    # Yen のアルゴリズムで一時的に禁止しているエッジ
    if (edge.source, edge.target) in blocked_edges:
        return False
    # 「この駅で乗り換えられなかったら？」で無効化された乗換エッジ
    if edge.kind == "transfer" and edge.transfer_id in disabled_transfers:
        return False
    # 「区間閉鎖モード」で閉じられた乗車エッジ
    if edge.kind == "ride" and edge.segment_id in closed_segments:
        return False
    return True


def dijkstra_shortest_path(
    graph: MetroGraph,
    start_nodes: Sequence[str],
    goal_nodes: Set[str],
    priority: str = "fewest_transfers",
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
    blocked_nodes: Optional[Set[str]] = None,
    blocked_edges: Optional[Set[Tuple[str, str]]] = None,
    blocked_station_names: Optional[Set[str]] = None,
    forbid_initial_transfer: bool = False,
) -> Optional[Tuple[List[str], List[Edge], int, int]]:
    """拡張状態つきのダイクストラ法で最短経路を 1 本探索する.

    【ポイント】
    探索状態を単なる `node_id` ではなく `(node_id, last_was_transfer)` のペアにしています。
    これは「電車に乗らずに 2 回連続で乗換エッジを辿る（A線→B線→A線のような無意味な往復）」
    を防ぐためです。
    """
    disabled_tr = set(disabled_transfers or ())
    closed_seg = set(closed_segments or ())
    b_nodes = set(blocked_nodes or ())
    b_edges = set(blocked_edges or ())
    b_stations = set(blocked_station_names or ())

    # 優先度付きキュー（最小ヒープ）の要素:
    # (第1コスト, 第2コスト, タイブレーク連番, ノードID, 直前が乗換か, 累積乗車区間数, 累積乗換回数)
    pq: List[Tuple[int, int, int, str, bool, int, int]] = []
    # 各状態 (node_id, last_was_transfer) に到達した最小コストを記録
    best: Dict[Tuple[str, bool], Tuple[int, int]] = {}
    # 経路復元用の親ポインタ: 状態 -> (前の状態, 使ったエッジ)
    parent: Dict[Tuple[str, bool], Tuple[Optional[Tuple[str, bool]], Optional[Edge]]] = {}

    counter = 0
    # 出発駅に複数路線が乗り入れている場合（例: 銀座 -> G-09 と M-16）、
    # 全ホームをコスト (0, 0) の初期候補としてキューに投入する
    for s_node in start_nodes:
        if s_node in b_nodes:
            continue
        state = (s_node, forbid_initial_transfer)
        init_cost = (0, 0)
        best[state] = init_cost
        parent[state] = (None, None)
        heapq.heappush(pq, (init_cost[0], init_cost[1], counter, s_node, forbid_initial_transfer, 0, 0))
        counter += 1

    while pq:
        c0, c1, _, curr_node, last_was_transfer, stops, transfers = heapq.heappop(pq)
        curr_state = (curr_node, last_was_transfer)

        # より小さいコストで既に確定済みの古いエントリならスキップ
        if best.get(curr_state) != (c0, c1):
            continue

        # 目的駅のいずれかの路線ノードに到達した瞬間が最短確定
        if curr_node in goal_nodes:
            # parent ポインタを逆順に辿って経路（ノード列・エッジ列）を復元する
            path_nodes: List[str] = []
            path_edges: List[Edge] = []
            walk: Optional[Tuple[str, bool]] = curr_state
            while walk is not None:
                path_nodes.append(walk[0])
                prev_state, edge = parent[walk]
                if edge is not None:
                    path_edges.append(edge)
                walk = prev_state
            path_nodes.reverse()
            path_edges.reverse()
            return path_nodes, path_edges, stops, transfers

        # 現在ノードから出ている全エッジを展開
        for edge in graph.adjacency.get(curr_node, []):
            nxt = edge.target
            if nxt in b_nodes:
                continue
            if not _edge_allowed(edge, disabled_tr, closed_seg, b_edges):
                continue

            # 連続した乗換エッジの使用を禁止（列車に乗らずに乗り換え直すのを防ぐ）
            if edge.kind == "transfer" and last_was_transfer:
                continue

            # Yen の分岐探索（spur search）で既に通過済みの「駅名」に再侵入するループを防ぐ
            nxt_station = graph.nodes[nxt].station_name
            curr_station = graph.nodes[curr_node].station_name
            if nxt_station in b_stations and nxt_station != curr_station and nxt not in goal_nodes:
                continue

            # 新しいコストを計算し、過去最良を更新できた場合のみキューに追加
            new_stops = stops + (1 if edge.kind == "ride" else 0)
            new_transfers = transfers + (1 if edge.kind == "transfer" else 0)
            new_cost = _cost_tuple(new_stops, new_transfers, priority)
            nxt_state = (nxt, edge.kind == "transfer")

            prev_best = best.get(nxt_state)
            if prev_best is None or new_cost < prev_best:
                best[nxt_state] = new_cost
                parent[nxt_state] = (curr_state, edge)
                heapq.heappush(
                    pq,
                    (new_cost[0], new_cost[1], counter, nxt, edge.kind == "transfer", new_stops, new_transfers),
                )
                counter += 1

    # 制約により到達可能な経路が 1 本もない場合
    return None


def _is_realistic_simple_path(graph: MetroGraph, node_ids: List[str], edges: List[Edge]) -> bool:
    """経路が現実的な単純経路かチェックする（同じ駅への再訪や無意味な始終端乗換を除外）.

    例として以下のような不自然な候補を弾きます:
      - 一度離れた駅に別路線で戻ってくるループ経路
      - 出発駅や到着駅のホーム内で、電車に乗る前後に不要な乗換を行う経路
      - 2 回連続の乗換
    """
    # 同じノードを 2 回踏んでいれば NG
    if len(set(node_ids)) != len(node_ids):
        return False

    # 始端・終端で「同一駅名内の乗換」から始まる／終わる無意味な経路を除外
    # （異駅名乗換：溜池山王 <-> 国会議事堂前 の徒歩移動は許可する）
    if edges and edges[0].kind == "transfer":
        if graph.nodes[node_ids[0]].station_name == graph.nodes[node_ids[1]].station_name:
            return False
    if edges and edges[-1].kind == "transfer":
        if graph.nodes[node_ids[-2]].station_name == graph.nodes[node_ids[-1]].station_name:
            return False

    # 駅名レベルでの再訪チェック（乗換によって同一駅名ノードが連続するのは OK だが、一度別駅を挟んで戻るのは NG）
    visited_stations: Set[str] = set()
    prev_station: Optional[str] = None
    for idx, nid in enumerate(node_ids):
        st_name = graph.nodes[nid].station_name
        if st_name != prev_station:
            if st_name in visited_stations:
                return False
            visited_stations.add(st_name)
        if idx > 0 and edges[idx - 1].kind == "transfer":
            # 2 回連続の乗換がないことも念のため確認
            if idx >= 2 and edges[idx - 2].kind == "transfer":
                return False
        prev_station = st_name
    return True


def yen_k_shortest_paths(
    graph: MetroGraph,
    origin_station: str,
    destination_station: str,
    priority: str = "fewest_transfers",
    k: int = 5,
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
) -> List[Tuple[List[str], List[Edge], int, int]]:
    """Yen の K 最短単純経路アルゴリズムで、上位 k 件の異なる経路候補を探索する.

    【アルゴリズムの概要】
    1. まずダイクストラ法で第 1 候補（最短経路）を求める。
    2. 直前に確定した経路の各ノードを「分岐点（spur node）」とみなし、
       既存候補と同じ前半（root path）を辿った直後のエッジを一時的に通行止めにして
       分岐点から目的地までの別ルート（spur path）をダイクストラ法で再探索する。
    3. 見つかった別ルートを候補ヒープ（potential）に溜め、最もコストの小さいものを次の確定候補に採用する。
    4. 確定候補が k 件集まるか、分岐候補が尽きるまで繰り返す。
    """
    # 出発駅と目的駅が同じ場合の特殊処理
    if origin_station == destination_station:
        start_node = graph.station_name_to_nodes[origin_station][0]
        return [([start_node], [], 0, 0)]

    start_nodes = graph.station_name_to_nodes[origin_station]
    goal_nodes = set(graph.station_name_to_nodes[destination_station])
    disabled_tr = set(disabled_transfers or ())
    closed_seg = set(closed_segments or ())

    # Step 1: 通常のダイクストラ法で第 1 候補を求める
    first = dijkstra_shortest_path(
        graph=graph,
        start_nodes=start_nodes,
        goal_nodes=goal_nodes,
        priority=priority,
        disabled_transfers=disabled_tr,
        closed_segments=closed_seg,
    )
    if first is None:
        return []

    accepted: List[Tuple[List[str], List[Edge], int, int]] = [first]
    accepted_sigs: Set[str] = {"->".join(first[0])}
    # 次に採用する候補をコスト順に管理する最小ヒープ:
    # (cost0, cost1, tie_idx, nodes, edges, stops, transfers)
    potential: List[Tuple[int, int, int, List[str], List[Edge], int, int]] = []
    potential_sigs: Set[str] = set()
    tie = 0

    # フィルタで落ちる候補も見越して最大 k*3 回までイテレーションする
    for _ in range(1, k * 3):
        if len(accepted) >= k:
            break
        prev_nodes, prev_edges, _, _ = accepted[-1]

        # Step 2: 直前の確定経路の各位置を分岐点として迂回候補を探す
        for spur_idx in range(len(prev_nodes) - 1):
            spur_node = prev_nodes[spur_idx]
            root_nodes = prev_nodes[: spur_idx + 1]
            root_edges = prev_edges[:spur_idx]

            # これまでに確定した経路のうち、同じ root_nodes を共有するものが次に進んだエッジを禁止
            blocked_edges: Set[Tuple[str, str]] = set()
            for acc_nodes, _, _, _ in accepted:
                if len(acc_nodes) > spur_idx + 1 and acc_nodes[: spur_idx + 1] == root_nodes:
                    blocked_edges.add((acc_nodes[spur_idx], acc_nodes[spur_idx + 1]))

            # root 部分で通ったノード・駅名へ戻らないようにブロック
            blocked_nodes = set(root_nodes[:-1])
            blocked_stations = {graph.nodes[nid].station_name for nid in root_nodes[:-1]}
            # root の末尾が乗換エッジなら、分岐直後にさらに乗換を連続させない
            forbid_initial_transfer = bool(root_edges and root_edges[-1].kind == "transfer")

            spur_res = dijkstra_shortest_path(
                graph=graph,
                start_nodes=[spur_node],
                goal_nodes=goal_nodes,
                priority=priority,
                disabled_transfers=disabled_tr,
                closed_segments=closed_seg,
                blocked_nodes=blocked_nodes,
                blocked_edges=blocked_edges,
                blocked_station_names=blocked_stations,
                forbid_initial_transfer=forbid_initial_transfer,
            )
            if spur_res is None:
                continue

            # 前半 (root) と後半 (spur) を結合して完全な経路候補にする
            spur_nodes, spur_edges, _, _ = spur_res
            total_nodes = root_nodes[:-1] + spur_nodes
            total_edges = root_edges + spur_edges

            if not _is_realistic_simple_path(graph, total_nodes, total_edges):
                continue

            sig = "->".join(total_nodes)
            if sig in accepted_sigs or sig in potential_sigs:
                continue

            total_stops = sum(1 for e in total_edges if e.kind == "ride")
            total_transfers = sum(1 for e in total_edges if e.kind == "transfer")
            c0, c1 = _cost_tuple(total_stops, total_transfers, priority)
            heapq.heappush(
                potential,
                (c0, c1, tie, total_nodes, total_edges, total_stops, total_transfers),
            )
            potential_sigs.add(sig)
            tie += 1

        if not potential:
            break

        # Step 3: 候補ヒープから最良の 1 本を取り出して確定リストに追加
        _, _, _, best_nodes, best_edges, best_stops, best_transfers = heapq.heappop(potential)
        best_sig = "->".join(best_nodes)
        if best_sig not in accepted_sigs:
            accepted.append((best_nodes, best_edges, best_stops, best_transfers))
            accepted_sigs.add(best_sig)

    return accepted


def _build_route_candidate(
    graph: MetroGraph,
    origin_station: str,
    destination_station: str,
    node_ids: List[str],
    edges: List[Edge],
    stops: int,
    transfers: int,
) -> RouteCandidate:
    """探索されたノード列・エッジ列を、人間が読みやすい RouteCandidate 構造へ変換する.

    乗車区間ごとのグループ化（legs）、乗換ステップの抽出（transfer_steps）、
    丸ノ内線・中野坂上駅での方南町支線折り返し判定、特徴量計算、日本語案内文生成までを行います。
    """
    legs: List[RouteLeg] = []
    transfer_steps: List[TransferStep] = []

    if not edges:
        # 同一駅（移動なし）の場合
        node = graph.nodes[node_ids[0]]
        line = graph.lines[node.line_id]
        legs.append(
            RouteLeg(
                line_id=line.id,
                line_name=line.name,
                direction_label="",
                node_ids=[node.id],
                station_names=[node.station_name],
                stops=0,
            )
        )
    else:
        curr_leg_nodes: List[str] = [node_ids[0]]
        curr_line_id: Optional[str] = None
        curr_dir: Optional[str] = None
        curr_is_branch = False
        stops_so_far = 0

        def flush_leg() -> None:
            """現在蓄積中の乗車区間 (curr_leg_nodes) を 1 つの RouteLeg として確定する."""
            nonlocal curr_leg_nodes, curr_line_id, curr_dir, curr_is_branch
            if len(curr_leg_nodes) >= 2 and curr_line_id is not None:
                line = graph.lines[curr_line_id]
                # 方南町支線から中野坂上を越えて池袋方面へ直通する場合、最終到達駅に応じて方面名を自然化
                refined_dir = curr_dir or ""
                if curr_line_id == "M" and refined_dir == "中野坂上・池袋方面":
                    end_node = curr_leg_nodes[-1]
                    if end_node.startswith("M-") and int(end_node.split("-")[1]) >= 7:
                        refined_dir = "池袋方面"
                    else:
                        refined_dir = "中野坂上方面"
                legs.append(
                    RouteLeg(
                        line_id=line.id,
                        line_name=line.name,
                        direction_label=refined_dir,
                        node_ids=list(curr_leg_nodes),
                        station_names=[graph.nodes[nid].station_name for nid in curr_leg_nodes],
                        stops=len(curr_leg_nodes) - 1,
                        is_branch_leg=curr_is_branch,
                    )
                )

        for idx, edge in enumerate(edges):
            if edge.kind == "ride":
                src_node = graph.nodes[edge.source]
                dst_node = graph.nodes[edge.target]

                # 丸ノ内線の中野坂上 (M-06) で支線と荻窪方面を折り返す動きを検出
                # 例: 中野新橋 (Mb-05) -> 中野坂上 (M-06) -> 新中野 (M-05)（および逆方向）
                # 路線自体は同じ丸ノ内線なので transfer=0 のままだが、案内上は 2 つの leg に分ける
                branch_turn = False
                if curr_line_id == "M" and len(curr_leg_nodes) >= 2:
                    prev_node_id = curr_leg_nodes[-2]
                    if edge.source == "M-06":
                        if (prev_node_id == "Mb-05" and edge.target == "M-05") or (
                            prev_node_id == "M-05" and edge.target == "Mb-05"
                        ):
                            branch_turn = True

                if branch_turn:
                    flush_leg()
                    curr_leg_nodes = [edge.source, edge.target]
                    curr_line_id = edge.line_id
                    curr_dir = edge.direction_label
                    curr_is_branch = src_node.branch == "honancho" or dst_node.branch == "honancho"
                else:
                    if curr_line_id is None:
                        curr_line_id = edge.line_id
                        curr_dir = edge.direction_label
                        curr_is_branch = src_node.branch == "honancho" or dst_node.branch == "honancho"
                    else:
                        # 本線から方南町支線へ直通分岐した場合／支線から本線へ合流した場合に方面名を更新
                        if edge.direction_label == "方南町方面":
                            curr_dir = "方南町方面"
                        elif curr_dir == "中野坂上・池袋方面" and edge.direction_label == "池袋方面":
                            curr_dir = "池袋方面"
                        if src_node.branch == "honancho" or dst_node.branch == "honancho":
                            curr_is_branch = True
                    curr_leg_nodes.append(edge.target)
                stops_so_far += 1
            else:
                # 乗換エッジに到達したら、直前までの乗車区間を確定して TransferStep を記録する
                flush_leg()
                src_node = graph.nodes[edge.source]
                dst_node = graph.nodes[edge.target]
                transfer_steps.append(
                    TransferStep(
                        transfer_id=edge.transfer_id or "",
                        transfer_label=edge.transfer_label or src_node.station_name,
                        from_node_id=edge.source,
                        to_node_id=edge.target,
                        from_station=src_node.station_name,
                        to_station=dst_node.station_name,
                        from_line_id=src_node.line_id,
                        to_line_id=dst_node.line_id,
                        is_diff_name=edge.is_diff_name,
                        stop_index_before=stops_so_far,
                    )
                )
                curr_leg_nodes = [edge.target]
                curr_line_id = None
                curr_dir = None
                curr_is_branch = False

        # 最後の乗車区間を確定
        flush_leg()

    # PyTorch 好み学習用の追加特徴量を計算
    diff_name_transfers = sum(1 for t in transfer_steps if t.is_diff_name)
    if transfers > 0 and stops > 0 and transfer_steps:
        # 乗換が全行程のどの位置（0.0〜1.0）で起きるかの平均を取り、中心を 0 に揃える（-0.5〜+0.5）
        avg_pos = sum(t.stop_index_before / stops for t in transfer_steps) / len(transfer_steps)
        transfer_timing = avg_pos - 0.5
    else:
        transfer_timing = 0.0

    # 循環インポートを避けるため関数内で formatter をインポート
    from src.formatter import format_route_guidance, format_route_title

    candidate = RouteCandidate(
        origin_station=origin_station,
        destination_station=destination_station,
        node_ids=node_ids,
        edges=edges,
        stops=stops,
        transfers=transfers,
        diff_name_transfers=diff_name_transfers,
        transfer_timing=transfer_timing,
        legs=legs,
        transfer_steps=transfer_steps,
    )
    candidate.title = format_route_title(candidate)
    candidate.guidance = format_route_guidance(candidate)
    return candidate


def search_routes(
    graph: MetroGraph,
    origin_station: str,
    destination_station: str,
    priority: str = "fewest_transfers",
    max_candidates: int = 5,
    disabled_transfers: Optional[Iterable[str]] = None,
    closed_segments: Optional[Iterable[str]] = None,
) -> List[RouteCandidate]:
    """UI や学習モジュールから呼び出すメインの経路検索エントリーポイント.

    【工夫している点】
    指定された priority だけで探索すると、たとえば「乗換 0 回・8 駅」と「乗換 1 回・7 駅」の
    トレードオフ候補が片方しか上位に出ないことがあります。
    そこで内部では必ず 'fewest_transfers' と 'fewest_stops' の **両方の基準** で Yen 探索を実行し、
    重複を除いたプールを作ってから要求された priority で並べ替えています。
    """
    if origin_station not in graph.station_name_to_nodes:
        raise ValueError(f"Unknown origin station: {origin_station}")
    if destination_station not in graph.station_name_to_nodes:
        raise ValueError(f"Unknown destination station: {destination_station}")

    disabled_tr = set(disabled_transfers or ())
    closed_seg = set(closed_segments or ())

    # 両方の評価基準で候補を集めてパレート最適なトレードオフ候補を取りこぼさないようにする
    raw_pool: Dict[str, Tuple[List[str], List[Edge], int, int]] = {}
    for mode in ("fewest_transfers", "fewest_stops"):
        paths = yen_k_shortest_paths(
            graph=graph,
            origin_station=origin_station,
            destination_station=destination_station,
            priority=mode,
            k=max(max_candidates, 5),
            disabled_transfers=disabled_tr,
            closed_segments=closed_seg,
        )
        for item in paths:
            sig = "->".join(item[0])
            if sig not in raw_pool:
                raw_pool[sig] = item

    if not raw_pool:
        return []

    candidates = [
        _build_route_candidate(graph, origin_station, destination_station, n, e, s, t)
        for (n, e, s, t) in raw_pool.values()
    ]

    # 最短駅数より 8 駅超の極端な大回りや、最小乗換 + 2 回超の非現実的な迂回を足切りする
    min_stops = min(c.stops for c in candidates)
    min_transfers = min(c.transfers for c in candidates)
    filtered = [
        c
        for c in candidates
        if c.stops <= min_stops + 8 and c.transfers <= min_transfers + 2
    ]
    if not filtered:
        filtered = candidates

    # 要求された優先基準（同値なら異駅名乗換が少ない順→タイトル順）で安定ソート
    if priority == "fewest_stops":
        filtered.sort(key=lambda c: (c.stops, c.transfers, c.diff_name_transfers, c.title))
    else:
        filtered.sort(key=lambda c: (c.transfers, c.stops, c.diff_name_transfers, c.title))

    return filtered[:max_candidates]
