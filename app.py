"""Streamlit App: 東京メトロ 乗換シミュレーター（MVP: 銀座線＋丸ノ内線）."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

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

st.set_page_config(
    page_title="東京メトロ 乗換シミュレーター",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Subtle custom CSS for clean card presentation
st.markdown(
    """
    <style>
    .route-card {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        background: #FFFFFF;
    }
    .route-card-selected {
        border: 2px solid #0F172A;
        background: #F8FAFC;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-g { background: #FF9500; color: #FFFFFF; }
    .badge-m { background: #F62E36; color: #FFFFFF; }
    .badge-stat { background: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; }
    .badge-score { background: #EDE9FE; color: #5B21B6; border: 1px solid #C4B5FD; }
    .diff-box {
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #CBD5E1;
        background: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph() -> MetroGraph:
    return MetroGraph.from_json()


graph = get_graph()
station_names = graph.get_station_names()
transfers_list = graph.get_ordered_transfers()
segments_list = graph.get_ordered_segments()
segment_id_to_label = {s.id: s.label for s in segments_list}
transfer_id_to_label = {t.id: t.label for t in transfers_list}


def init_state() -> None:
    defaults = {
        "origin_select": "浅草",
        "destination_select": "新宿",
        "priority_select": "fewest_transfers",
        "disabled_transfers": [],
        "closed_segments_select": [],
        "selected_candidate_idx": 0,
        "last_loss_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    for tr in transfers_list:
        chk_key = f"chk_{tr.id}"
        if chk_key not in st.session_state:
            st.session_state[chk_key] = tr.id in st.session_state.disabled_transfers


init_state()


def apply_preset(
    origin: str,
    destination: str,
    priority: str = "fewest_transfers",
    disabled_transfers: Optional[List[str]] = None,
    closed_segments: Optional[List[str]] = None,
) -> None:
    dis_list = list(disabled_transfers or [])
    cls_list = list(closed_segments or [])
    st.session_state.origin_select = origin
    st.session_state.destination_select = destination
    st.session_state.priority_select = priority
    st.session_state.disabled_transfers = dis_list
    st.session_state.closed_segments_select = cls_list
    for tr in transfers_list:
        st.session_state[f"chk_{tr.id}"] = tr.id in dis_list
    st.session_state.selected_candidate_idx = 0


def swap_stations() -> None:
    st.session_state.origin_select, st.session_state.destination_select = (
        st.session_state.destination_select,
        st.session_state.origin_select,
    )
    st.session_state.selected_candidate_idx = 0


def reset_constraints() -> None:
    st.session_state.disabled_transfers = []
    st.session_state.closed_segments_select = []
    for tr in transfers_list:
        st.session_state[f"chk_{tr.id}"] = False
    st.session_state.selected_candidate_idx = 0


def reset_ml_state() -> None:
    clear_choice_logs(DEFAULT_LOG_PATH, DEFAULT_MODEL_PATH)
    st.session_state.last_loss_history = []
    st.session_state.priority_select = "fewest_transfers"


# Header
st.title("🚇 東京メトロ 乗換シミュレーター")
st.caption(
    "MVP（銀座線・丸ノ内線＋方南町支線）｜「この駅で乗り換えられなかったら？」「区間が止まったら？」を路線図上で実験し、PyTorchで自分の経路の好みを学習します。"
)

# Preset scenarios bar
st.markdown("**クイック体験シナリオ**")
p_cols = st.columns(5)
with p_cols[0]:
    st.button(
        "浅草→新宿（通常）",
        width="stretch",
        on_click=apply_preset,
        args=("浅草", "新宿", "fewest_transfers", [], []),
    )
with p_cols[1]:
    st.button(
        "浅草→新宿（銀座乗換NG）",
        width="stretch",
        on_click=apply_preset,
        args=("浅草", "新宿", "fewest_transfers", ["transfer:銀座"], []),
    )
with p_cols[2]:
    st.button(
        "渋谷→銀座（0回 vs 短縮）",
        width="stretch",
        on_click=apply_preset,
        args=("渋谷", "銀座", "fewest_stops", [], []),
    )
with p_cols[3]:
    st.button(
        "方南町→銀座（支線直通）",
        width="stretch",
        on_click=apply_preset,
        args=("方南町", "銀座", "fewest_transfers", [], []),
    )
with p_cols[4]:
    st.button(
        "新宿→東京（霞ケ関手前閉鎖）",
        width="stretch",
        on_click=apply_preset,
        args=("新宿", "東京", "fewest_transfers", [], ["M-14__M-15"]),
    )

tab_sim, tab_ml = st.tabs(["🗺️ 乗換シミュレーター", "🧠 PyTorch 好み学習 & Colab連携"])

saved_model_bundle = load_saved_model(DEFAULT_MODEL_PATH)
trained_model = saved_model_bundle[0] if saved_model_bundle else None
trained_summary = saved_model_bundle[1] if saved_model_bundle else None

# ==============================================================================
# TAB 1: ROUTE SIMULATOR
# ==============================================================================
with tab_sim:
    # Top search controls
    c1, c_swap, c2, c3 = st.columns([3, 0.7, 3, 3.2])
    with c1:
        origin = st.selectbox(
            "出発駅",
            options=station_names,
            format_func=graph.get_station_display_label,
            key="origin_select",
        )
    with c_swap:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("⇄", help="出発駅と目的駅を入れ替え", width="stretch", on_click=swap_stations)
    with c2:
        destination = st.selectbox(
            "目的駅",
            options=station_names,
            format_func=graph.get_station_display_label,
            key="destination_select",
        )
    with c3:
        priority_options = ["fewest_transfers", "fewest_stops"]
        if trained_model is not None:
            priority_options.append("learned")
        if st.session_state.priority_select not in priority_options:
            st.session_state.priority_select = "fewest_transfers"
        priority_labels = {
            "fewest_transfers": "乗換が少ない順（ダイクストラ）",
            "fewest_stops": "乗車区間数が少ない順（ダイクストラ）",
            "learned": "✨ PyTorch 好み学習順（再ランキング）",
        }
        priority = st.selectbox(
            "優先条件",
            options=priority_options,
            format_func=lambda p: priority_labels[p],
            key="priority_select",
        )

    # Disruption controls ("この駅で乗り換えられなかったら？" & 区間閉鎖)
    with st.container(border=True):
        ctrl_left, ctrl_right, ctrl_reset = st.columns([4.5, 4.5, 1.3])
        with ctrl_left:
            st.markdown("**🚫 この駅で乗り換えられなかったら？（乗換無効化）**")
            st.caption("乗換のみ無効にします。その駅を通過する列車は引き続き利用できます。")
            new_disabled: List[str] = []
            tr_cols = st.columns(len(transfers_list))
            for col, tr in zip(tr_cols, transfers_list):
                with col:
                    checked = st.checkbox(
                        tr.label,
                        key=f"chk_{tr.id}",
                        help=tr.note,
                    )
                    if checked:
                        new_disabled.append(tr.id)
            st.session_state.disabled_transfers = new_disabled

        with ctrl_right:
            st.markdown("**🚧 区間そのものを閉じるモード（運休シミュレーション）**")
            st.caption("選択した隣駅間の線路（双方向）を閉鎖して迂回経路を調べます。")
            closed_selected = st.multiselect(
                "閉鎖する区間",
                options=[s.id for s in segments_list],
                format_func=lambda sid: segment_id_to_label[sid],
                label_visibility="collapsed",
                placeholder="閉鎖したい区間を選択（複数可）...",
                key="closed_segments_select",
            )

        with ctrl_reset:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            st.button("制約を解除", width="stretch", on_click=reset_constraints)

    # Run routing (baseline without disruptions + active with current disruptions)
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
        disabled_transfers=st.session_state.disabled_transfers,
        closed_segments=closed_selected,
    )

    if priority == "learned" and trained_model is not None:
        baseline_candidates = rerank_candidates(baseline_candidates, trained_model)
        active_candidates = rerank_candidates(active_candidates, trained_model)

    if st.session_state.selected_candidate_idx >= len(active_candidates):
        st.session_state.selected_candidate_idx = 0

    selected_route: Optional[RouteCandidate] = (
        active_candidates[st.session_state.selected_candidate_idx] if active_candidates else None
    )
    baseline_top: Optional[RouteCandidate] = baseline_candidates[0] if baseline_candidates else None
    has_constraints = bool(st.session_state.disabled_transfers or closed_selected)

    # Main 2-column visualization + candidates panel
    map_col, list_col = st.columns([6.3, 3.7], gap="large")

    with map_col:
        svg_html = render_metro_svg(
            graph=graph,
            active_route=selected_route,
            baseline_route=baseline_top if has_constraints else None,
            disabled_transfers=st.session_state.disabled_transfers,
            closed_segments=closed_selected,
            origin_station=origin,
            destination_station=destination,
        )
        st.markdown(svg_html, unsafe_allow_html=True)

        # Before vs After Comparison Section (shown whenever constraints are active)
        if has_constraints:
            st.markdown("#### 🔄 変更前後の比較（どの乗換駅・迂回ルートが代わりを担えるか）")
            comp_left, comp_right = st.columns(2)
            with comp_left:
                st.markdown("**変更前（制約なし・第1候補）**")
                if baseline_top:
                    st.markdown(
                        f"""<div class="diff-box">
                        <b>{baseline_top.title}</b><br/>
                        <span class="badge badge-stat">乗換 {baseline_top.transfers}回</span>
                        <span class="badge badge-stat">乗車 {baseline_top.stops}区間</span><br/>
                        <div style="margin-top:6px;font-size:0.9rem;color:#334155;">{baseline_top.guidance}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("変更前の候補はありません。")
            with comp_right:
                st.markdown("**変更後（現在の制約・第1候補）**")
                if active_candidates:
                    new_top = active_candidates[0]
                    if baseline_top and new_top.signature == baseline_top.signature:
                        delta_html = '<span style="color:#059669;font-weight:700;">変化なし（制約の影響を受けない経路です）</span>'
                    elif baseline_top:
                        d_tr = new_top.transfers - baseline_top.transfers
                        d_st = new_top.stops - baseline_top.stops
                        d_tr_str = f"{d_tr:+d}回"
                        d_st_str = f"{d_st:+d}区間"
                        delta_html = (
                            f'<span style="color:#B45309;font-weight:700;">'
                            f"代替経路に切替（乗換 {d_tr_str} / 乗車 {d_st_str}）</span>"
                        )
                    else:
                        delta_html = ""
                    st.markdown(
                        f"""<div class="diff-box" style="border-color:#F59E0B;background:#FFFBEB;">
                        <b>{new_top.title}</b> {delta_html}<br/>
                        <span class="badge badge-stat">乗換 {new_top.transfers}回</span>
                        <span class="badge badge-stat">乗車 {new_top.stops}区間</span><br/>
                        <div style="margin-top:6px;font-size:0.9rem;color:#334155;">{new_top.guidance}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("到達可能な経路がありません（すべての接続が遮断されています）。")

            # Show dropped routes summary
            active_sigs = {c.signature for c in active_candidates}
            dropped = [c for c in baseline_candidates if c.signature not in active_sigs]
            if dropped:
                dropped_names = "、".join(c.title for c in dropped)
                st.caption(f"除外された候補: {dropped_names}")

    with list_col:
        st.subheader(f"経路候補（{origin} → {destination}）")
        if origin == destination:
            st.info("出発駅と目的駅が同じです。別の駅を選んでください。")
        elif not active_candidates:
            st.error(
                "指定した条件では目的駅に到達できません。無効化した乗換または閉鎖区間を一部解除してみてください。"
            )
        else:
            # Summary comparison table matching design spec
            summary_rows = [
                {
                    "順位": f"#{i + 1}",
                    "候補": c.title,
                    "乗換": f"{c.transfers}回",
                    "乗車": f"{c.stops}区間",
                    **(
                        {"好みスコア": f"{c.score:+.2f}"}
                        if priority == "learned" and c.score is not None
                        else {}
                    ),
                }
                for i, c in enumerate(active_candidates)
            ]
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

            # Detailed candidate cards with preview & "私ならこの経路を選ぶ" button
            for idx, cand in enumerate(active_candidates):
                is_sel = idx == st.session_state.selected_candidate_idx
                card_cls = "route-card route-card-selected" if is_sel else "route-card"

                leg_badges = ""
                for leg in cand.legs:
                    cls = "badge-g" if leg.line_id == "G" else "badge-m"
                    leg_badges += (
                        f'<span class="badge {cls}">{leg.line_id} {leg.line_name} '
                        f"({leg.stops}区間)</span>"
                    )
                score_badge = (
                    f'<span class="badge badge-score">好み {cand.score:+.2f}</span>'
                    if priority == "learned" and cand.score is not None
                    else ""
                )
                diff_badge = (
                    '<span class="badge badge-stat">異駅名乗換あり</span>'
                    if cand.diff_name_transfers > 0
                    else ""
                )

                st.markdown(
                    f"""<div class="{card_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                      <span style="font-weight:800;font-size:1.02rem;color:#0F172A;">
                        {"▶ " if is_sel else ""}#{idx + 1} {cand.title}
                      </span>
                      <span>
                        <span class="badge badge-stat">乗換 {cand.transfers}回</span>
                        <span class="badge badge-stat">乗車 {cand.stops}区間</span>
                        {score_badge}
                      </span>
                    </div>
                    <div style="margin:6px 0;">{leg_badges}{diff_badge}</div>
                    <div style="font-size:0.92rem;color:#1E293B;line-height:1.5;">{cand.guidance}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                b_col1, b_col2 = st.columns([1, 1.35])
                with b_col1:
                    if st.button(
                        "路線図に表示中" if is_sel else "路線図で確認",
                        key=f"preview_{idx}_{cand.signature}",
                        disabled=is_sel,
                        width="stretch",
                    ):
                        st.session_state.selected_candidate_idx = idx
                        st.rerun()
                with b_col2:
                    if st.button(
                        "👍 私ならこの経路を選ぶ",
                        key=f"choose_{idx}_{cand.signature}",
                        type="primary" if is_sel else "secondary",
                        width="stretch",
                    ):
                        append_choice_log(
                            origin=origin,
                            destination=destination,
                            priority_mode=priority,
                            disabled_transfers=st.session_state.disabled_transfers,
                            closed_segments=closed_selected,
                            candidates=active_candidates,
                            chosen_index=idx,
                            log_path=DEFAULT_LOG_PATH,
                        )
                        st.toast(f"選択を記録しました：{cand.title}（{origin}→{destination}）", icon="✅")

                with st.expander("停車駅・詳細を見る", expanded=False):
                    for leg in cand.legs:
                        st.markdown(
                            f"- **{leg.line_name}（{leg.direction_label}）**：{' → '.join(leg.station_names)}"
                        )
                    if cand.transfer_steps:
                        for t in cand.transfer_steps:
                            tag = "（異駅名・連絡通路）" if t.is_diff_name else ""
                            st.markdown(
                                f"- **乗換**：{t.transfer_label}{tag}"
                            )


# ==============================================================================
# TAB 2: PYTORCH PREFERENCE LEARNING & COLAB EXPORT
# ==============================================================================
with tab_ml:
    st.subheader("🧠 PyTorchで自分の経路の好みを学習する")
    st.markdown(
        """
        シミュレーター画面で **「👍 私ならこの経路を選ぶ」** を押すと、その時表示されていた候補一覧と選択結果が
        `data/user_choices.jsonl` に蓄積されます。PyTorchの線形効用モデル（Multinomial Logit / Listwise Cross-Entropy）を
        初期重み `0` から学習し、**「乗換を1回減らすためなら何駅分の遠回りを許容するか」** を推定して候補を並べ替えます。
        """
    )

    logs = load_choice_logs(DEFAULT_LOG_PATH)
    usable_logs = [r for r in logs if len(r.get("candidates", [])) >= 2]

    m1, m2, m3 = st.columns(3)
    m1.metric("記録済み選択ログ", f"{len(logs)} 件")
    m2.metric("学習に使えるログ（2候補以上）", f"{len(usable_logs)} 件")
    if trained_summary and trained_summary.get("transfer_equivalent_stops") is not None:
        m3.metric(
            "学習済み：乗換1回の許容遠回り",
            f"約 {trained_summary['transfer_equivalent_stops']:.1f} 駅分",
        )
    else:
        m3.metric("学習済みモデル", "学習済み" if trained_model else "未学習")

    act1, act2, act3, act4 = st.columns([2.2, 2.2, 2.2, 1.4])
    with act1:
        if st.button("🚀 今すぐPyTorchで学習する", type="primary", width="stretch", disabled=len(usable_logs) == 0):
            model, summary = train_preference_model(
                records=logs,
                epochs=250,
                lr=0.08,
                weight_decay=0.02,
                save_path=DEFAULT_MODEL_PATH,
            )
            st.session_state.last_loss_history = summary.loss_history
            st.success(
                f"学習完了！（Loss: {summary.initial_loss:.4f} → {summary.final_loss:.4f} / "
                f"Top-1一致率: {summary.top1_accuracy * 100:.1f}%）"
            )
            st.rerun()

    with act2:
        if st.button("🧪 サンプル投入（乗換回避派）", width="stretch"):
            added = generate_sample_choice_logs(graph, persona="avoid_transfers", log_path=DEFAULT_LOG_PATH)
            st.toast(f"乗換回避派のサンプル選択を {len(added)} 件追加しました。", icon="🧪")
            st.rerun()

    with act3:
        if st.button("⚡ サンプル投入（駅数最短派）", width="stretch"):
            added = generate_sample_choice_logs(graph, persona="shortest_stops", log_path=DEFAULT_LOG_PATH)
            st.toast(f"駅数最短派のサンプル選択を {len(added)} 件追加しました。", icon="⚡")
            st.rerun()

    with act4:
        st.button("🗑️ リセット", width="stretch", on_click=reset_ml_state)

    # Show trained model analysis
    if trained_model is not None and trained_summary is not None:
        st.markdown("---")
        st.markdown("#### 📊 学習結果の解釈")
        st.info(f"**モデルの読み取り：** {trained_summary.get('interpretation', '')}")

        res_left, res_right = st.columns(2)
        with res_left:
            st.markdown("**学習された特徴量の重み（値が大きいほど好まれる・負ほど避けられる）**")
            weights_dict = trained_summary.get("weights", {})
            w_df = pd.DataFrame(
                {"特徴量": list(weights_dict.keys()), "重み": list(weights_dict.values())}
            ).set_index("特徴量")
            st.bar_chart(w_df)
            st.caption(
                f"学習イベント数: {trained_summary.get('num_events_used')}件 ｜ "
                f"Loss: {trained_summary.get('initial_loss')} → {trained_summary.get('final_loss')} ｜ "
                f"Top-1一致率: {trained_summary.get('top1_accuracy', 0) * 100:.1f}%"
            )

        with res_right:
            if st.session_state.last_loss_history:
                st.markdown("**学習Lossの推移（エポックごと）**")
                loss_df = pd.DataFrame({"Cross-Entropy Loss": st.session_state.last_loss_history})
                st.line_chart(loss_df)
            else:
                st.markdown("**並び順の変化チェック（標準 vs PyTorch学習後）**")
                st.caption("「今すぐPyTorchで学習する」を押すとLoss推移も表示されます。")

        # Side-by-side ranking comparison on representative pairs
        st.markdown("#### 🔀 学習前後の並び順比較（代表ルート）")
        compare_pairs = [
            ("浅草", "新宿"),
            ("渋谷", "銀座"),
            ("青山一丁目", "銀座"),
            ("渋谷", "東京"),
        ]
        comp_rows = []
        for o_st, d_st in compare_pairs:
            std_cands = search_routes(graph, o_st, d_st, priority="fewest_stops", max_candidates=4)
            learned_cands = rerank_candidates(std_cands, trained_model)
            if std_cands and learned_cands:
                comp_rows.append(
                    {
                        "区間": f"{o_st} → {d_st}",
                        "駅数最短順 1位": f"{std_cands[0].title}（乗換{std_cands[0].transfers}回 / {std_cands[0].stops}区間）",
                        "PyTorch学習後 1位": f"{learned_cands[0].title}（乗換{learned_cands[0].transfers}回 / {learned_cands[0].stops}区間 / スコア{learned_cands[0].score:+.2f}）",
                        "順位変化": "変化あり ✨" if std_cands[0].signature != learned_cands[0].signature else "同じ",
                    }
                )
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")

    # Export / inspect logs for Google Colab
    st.markdown("---")
    col_log_l, col_log_r = st.columns([3, 1.4])
    with col_log_l:
        st.markdown("#### 📁 選択ログ一覧 & Google Colab 連携")
        st.caption(
            "ダウンロードした `user_choices.jsonl` を `notebooks/tokyo_metro_preference_colab.ipynb` にアップロードすると、Colab上で学習・モデル比較・並び順変化の分析を実行できます。"
        )
    with col_log_r:
        if logs:
            jsonl_payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in logs) + "\n"
            st.download_button(
                label="⬇️ Colab用にJSONLをダウンロード",
                data=jsonl_payload.encode("utf-8"),
                file_name="user_choices.jsonl",
                mime="application/jsonl",
                width="stretch",
            )

    if logs:
        log_preview = [
            {
                "出発": r["origin"],
                "目的": r["destination"],
                "選んだ経路": r["chosen_title"],
                "候補数": len(r.get("candidates", [])),
                "無効乗換": ", ".join(r.get("disabled_transfers", [])) or "-",
            }
            for r in reversed(logs[-20:])
        ]
        st.dataframe(pd.DataFrame(log_preview), hide_index=True, width="stretch")
    else:
        st.info(
            "まだ選択ログがありません。「乗換シミュレーター」タブで「👍 私ならこの経路を選ぶ」を押すか、上のサンプル投入ボタンをお試しください。"
        )
