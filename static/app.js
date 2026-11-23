/**
 * 東京メトロ 乗換シミュレーター（FastAPI フロントエンド制御スクリプト）
 *
 * 画面全体の再読み込みを行わず、FastAPI の JSON API と非同期通信（fetch）して
 * SVG 路線図・経路候補カード・変更前後比較・PyTorch 学習結果をリアルタイムに更新します。
 */

// ============================================================================
// アプリ全体のフロントエンド状態（Streamlit の session_state に相当）
// ============================================================================
const state = {
  origin: "浅草",
  destination: "新宿",
  priority: "fewest_transfers",
  disabledTransfers: new Set(),
  closedSegments: [],
  selectedIndex: 0,
  hasTrainedModel: false,
  meta: {
    stations: [],
    transfers: [],
    segments: [],
    segmentMap: new Map(),
  },
};

// クイック体験シナリオのプリセット定義
const PRESETS = {
  normal: {
    origin: "浅草",
    destination: "新宿",
    priority: "fewest_transfers",
    disabledTransfers: [],
    closedSegments: [],
  },
  ginza_ng: {
    origin: "浅草",
    destination: "新宿",
    priority: "fewest_transfers",
    disabledTransfers: ["transfer:銀座"],
    closedSegments: [],
  },
  shibuya_ginza: {
    origin: "渋谷",
    destination: "銀座",
    priority: "fewest_stops",
    disabledTransfers: [],
    closedSegments: [],
  },
  honancho: {
    origin: "方南町",
    destination: "銀座",
    priority: "fewest_transfers",
    disabledTransfers: [],
    closedSegments: [],
  },
  segment_closed: {
    origin: "新宿",
    destination: "東京",
    priority: "fewest_transfers",
    disabledTransfers: [],
    closedSegments: ["M-14__M-15"],
  },
};

// ============================================================================
// ユーティリティ関数
// ============================================================================
function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showToast(message, isError = false) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = isError ? "toast toast-error" : "toast";
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.remove();
  }, 3400);
}

async function apiJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `API エラー (${res.status})`);
  }
  return data;
}

// ============================================================================
// UI 同期・コントロール描画
// ============================================================================
function syncPriorityOption(hasTrainedModel) {
  state.hasTrainedModel = hasTrainedModel;
  const learnedOpt = document.getElementById("learned-option");
  const prioritySelect = document.getElementById("priority-select");
  learnedOpt.hidden = !hasTrainedModel;
  if (!hasTrainedModel && state.priority === "learned") {
    state.priority = "fewest_transfers";
  }
  prioritySelect.value = state.priority;
}

function renderTransferCheckboxes() {
  const wrap = document.getElementById("transfer-checkboxes");
  wrap.innerHTML = "";

  for (const tr of state.meta.transfers) {
    const checked = state.disabledTransfers.has(tr.id);
    const label = document.createElement("label");
    label.className = `checkbox-pill${checked ? " checked" : ""}`;
    label.title = tr.note || "";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    input.addEventListener("change", () => {
      if (input.checked) {
        state.disabledTransfers.add(tr.id);
      } else {
        state.disabledTransfers.delete(tr.id);
      }
      state.selectedIndex = 0;
      renderTransferCheckboxes();
      runSearch();
    });

    const span = document.createElement("span");
    span.textContent = tr.label;
    label.append(input, span);
    wrap.appendChild(label);
  }
}

function renderClosedSegmentChips() {
  const wrap = document.getElementById("closed-segment-chips");
  wrap.innerHTML = "";
  for (const segId of state.closedSegments) {
    const seg = state.meta.segmentMap.get(segId);
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span>${escapeHtml(seg ? seg.label : segId)}</span>`;

    const rmBtn = document.createElement("button");
    rmBtn.type = "button";
    rmBtn.textContent = "✕";
    rmBtn.title = "閉鎖を解除";
    rmBtn.addEventListener("click", () => {
      state.closedSegments = state.closedSegments.filter((id) => id !== segId);
      state.selectedIndex = 0;
      renderClosedSegmentChips();
      runSearch();
    });
    chip.appendChild(rmBtn);
    wrap.appendChild(chip);
  }
}

function syncAllControls() {
  document.getElementById("origin-select").value = state.origin;
  document.getElementById("destination-select").value = state.destination;
  document.getElementById("priority-select").value = state.priority;
  renderTransferCheckboxes();
  renderClosedSegmentChips();
}

// ============================================================================
// タブ 1: 経路検索 API 呼び出しと画面描画
// ============================================================================
async function runSearch() {
  try {
    const data = await apiJson("/api/search", {
      method: "POST",
      body: JSON.stringify({
        origin: state.origin,
        destination: state.destination,
        priority: state.priority,
        disabled_transfers: Array.from(state.disabledTransfers),
        closed_segments: state.closedSegments,
        selected_index: state.selectedIndex,
      }),
    });

    state.selectedIndex = data.selected_index;
    syncPriorityOption(data.has_trained_model);

    // 1. SVG 路線図を差し替え
    document.getElementById("svg-container").innerHTML = data.svg;

    // 2. 変更前後比較セクションの描画
    renderComparison(data.comparison);

    // 3. 右カラムの経路候補テーブルと詳細カードの描画
    renderCandidates(data);
  } catch (err) {
    showToast(err.message, true);
  }
}

function renderComparison(comp) {
  const section = document.getElementById("comparison-section");
  if (!comp) {
    section.hidden = true;
    return;
  }
  section.hidden = false;

  const baseBox = document.getElementById("baseline-box");
  if (comp.baseline_top) {
    const b = comp.baseline_top;
    baseBox.innerHTML = `
      <div><strong>${escapeHtml(b.title)}</strong></div>
      <div class="card-badges">
        <span class="badge badge-stat">乗換 ${b.transfers}回</span>
        <span class="badge badge-stat">乗車 ${b.stops}区間</span>
      </div>
      <div>${escapeHtml(b.guidance)}</div>
    `;
  } else {
    baseBox.innerHTML = `<div>変更前の候補はありません。</div>`;
  }

  const activeBox = document.getElementById("active-box");
  if (comp.active_top) {
    const a = comp.active_top;
    let deltaHtml = "";
    if (comp.status === "unchanged") {
      deltaHtml = `<span class="delta-unchanged">変化なし（制約の影響を受けない経路です）</span>`;
    } else if (comp.status === "changed") {
      const dTr = `${comp.delta_transfers >= 0 ? "+" : ""}${comp.delta_transfers}回`;
      const dSt = `${comp.delta_stops >= 0 ? "+" : ""}${comp.delta_stops}区間`;
      deltaHtml = `<span class="delta-changed">代替経路に切替（乗換 ${dTr} / 乗車 ${dSt}）</span>`;
    }
    activeBox.innerHTML = `
      <div><strong>${escapeHtml(a.title)}</strong> ${deltaHtml}</div>
      <div class="card-badges">
        <span class="badge badge-stat">乗換 ${a.transfers}回</span>
        <span class="badge badge-stat">乗車 ${a.stops}区間</span>
      </div>
      <div>${escapeHtml(a.guidance)}</div>
    `;
  } else {
    activeBox.innerHTML = `<div style="color:#991b1b;font-weight:700;">到達可能な経路がありません（すべての接続が遮断されています）。</div>`;
  }

  const droppedEl = document.getElementById("dropped-caption");
  if (comp.dropped_titles && comp.dropped_titles.length > 0) {
    droppedEl.hidden = false;
    droppedEl.textContent = `除外された候補: ${comp.dropped_titles.join("、")}`;
  } else {
    droppedEl.hidden = true;
  }
}

function renderCandidates(data) {
  const heading = document.getElementById("candidates-heading");
  const banner = document.getElementById("status-banner");
  const tableWrap = document.getElementById("summary-table-wrap");
  const cardsEl = document.getElementById("candidate-cards");

  heading.textContent = `経路候補（${data.origin} → ${data.destination}）`;
  cardsEl.innerHTML = "";

  if (data.same_station) {
    banner.hidden = false;
    banner.className = "notice-box notice-info";
    banner.textContent = "出発駅と目的駅が同じです。別の駅を選んでください。";
    tableWrap.hidden = true;
    return;
  }

  if (!data.candidates || data.candidates.length === 0) {
    banner.hidden = false;
    banner.className = "notice-box notice-error";
    banner.textContent = "指定した条件では目的駅に到達できません。無効化した乗換または閉鎖区間を一部解除してみてください。";
    tableWrap.hidden = true;
    return;
  }

  banner.hidden = true;
  tableWrap.hidden = false;

  // サマリー比較テーブルの描画
  const showScore = data.priority === "learned" && data.candidates.some((c) => c.score !== null);
  const theadRow = document.getElementById("summary-thead-row");
  theadRow.innerHTML = `
    <th>順位</th>
    <th>候補</th>
    <th>乗換</th>
    <th>乗車</th>
    ${showScore ? "<th>好みスコア</th>" : ""}
  `;

  const tbody = document.getElementById("summary-tbody");
  tbody.innerHTML = data.candidates
    .map((c, idx) => {
      const scoreCell = showScore
        ? `<td>${c.score !== null ? (c.score >= 0 ? `+${c.score.toFixed(2)}` : c.score.toFixed(2)) : "-"}</td>`
        : "";
      return `
        <tr>
          <td>#${idx + 1}</td>
          <td>${escapeHtml(c.title)}</td>
          <td>${c.transfers}回</td>
          <td>${c.stops}区間</td>
          ${scoreCell}
        </tr>
      `;
    })
    .join("");

  // 候補カード一覧の描画
  data.candidates.forEach((cand, idx) => {
    const isSel = idx === data.selected_index;
    const card = document.createElement("article");
    card.className = `route-card${isSel ? " selected" : ""}`;

    const legBadges = cand.legs
      .map((leg) => {
        const cls = leg.line_id === "G" ? "badge-g" : "badge-m";
        return `<span class="badge ${cls}">${escapeHtml(leg.line_id)} ${escapeHtml(leg.line_name)} (${leg.stops}区間)</span>`;
      })
      .join("");

    const diffBadge =
      cand.diff_name_transfers > 0 ? `<span class="badge badge-stat">異駅名乗換あり</span>` : "";

    const scoreBadge =
      showScore && cand.score !== null
        ? `<span class="badge badge-score">好み ${cand.score >= 0 ? "+" : ""}${cand.score.toFixed(2)}</span>`
        : "";

    const detailItems = [
      ...cand.legs.map(
        (leg) =>
          `<li><strong>${escapeHtml(leg.line_name)}（${escapeHtml(leg.direction_label)}）</strong>：${escapeHtml(leg.station_names.join(" → "))}</li>`
      ),
      ...cand.transfer_steps.map((t) => {
        const tag = t.is_diff_name ? "（異駅名・連絡通路）" : "";
        return `<li><strong>乗換</strong>：${escapeHtml(t.transfer_label)}${tag}</li>`;
      }),
    ].join("");

    card.innerHTML = `
      <div class="card-top">
        <span class="card-title">${isSel ? "▶ " : ""}#${idx + 1} ${escapeHtml(cand.title)}</span>
        <div>
          <span class="badge badge-stat">乗換 ${cand.transfers}回</span>
          <span class="badge badge-stat">乗車 ${cand.stops}区間</span>
          ${scoreBadge}
        </div>
      </div>
      <div class="card-badges">${legBadges}${diffBadge}</div>
      <div class="card-guidance">${escapeHtml(cand.guidance)}</div>
      <div class="card-actions">
        <button type="button" class="btn btn-secondary btn-preview" ${isSel ? "disabled" : ""}>
          ${isSel ? "路線図に表示中" : "路線図で確認"}
        </button>
        <button type="button" class="btn ${isSel ? "btn-primary" : "btn-secondary"} btn-choose">
          👍 私ならこの経路を選ぶ
        </button>
      </div>
      <details class="card-details">
        <summary>停車駅・詳細を見る</summary>
        <ul>${detailItems}</ul>
      </details>
    `;

    // 「路線図で確認」クリック時は選択インデックスだけ変えて再検索（スクロール位置を維持したまま SVG を即更新）
    card.querySelector(".btn-preview").addEventListener("click", () => {
      state.selectedIndex = idx;
      runSearch();
    });

    // 「👍 私ならこの経路を選ぶ」クリック時は選択ログ API へ送信
    card.querySelector(".btn-choose").addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/choices", {
          method: "POST",
          body: JSON.stringify({
            origin: state.origin,
            destination: state.destination,
            priority: state.priority,
            disabled_transfers: Array.from(state.disabledTransfers),
            closed_segments: state.closedSegments,
            chosen_index: idx,
          }),
        });
        showToast(`✅ ${res.message}`);
        renderMlStatus(res.ml_status);
      } catch (err) {
        showToast(err.message, true);
      }
    });

    cardsEl.appendChild(card);
  });
}

// ============================================================================
// タブ 2: PyTorch 好み学習 & グラフ描画
// ============================================================================
async function loadMlStatus() {
  try {
    const status = await apiJson("/api/ml/status");
    renderMlStatus(status);
  } catch (err) {
    showToast(err.message, true);
  }
}

function renderMlStatus(status) {
  syncPriorityOption(status.has_trained_model);

  document.getElementById("metric-logs-total").textContent = `${status.logs_total} 件`;
  document.getElementById("metric-logs-usable").textContent = `${status.usable_logs} 件`;

  const modelLabel = document.getElementById("metric-model-label");
  const modelValue = document.getElementById("metric-model-value");
  if (status.trained_summary && status.trained_summary.transfer_equivalent_stops !== null && status.trained_summary.transfer_equivalent_stops !== undefined) {
    modelLabel.textContent = "学習済み：乗換1回の許容遠回り";
    modelValue.textContent = `約 ${Number(status.trained_summary.transfer_equivalent_stops).toFixed(1)} 駅分`;
  } else {
    modelLabel.textContent = "学習済みモデル";
    modelValue.textContent = status.has_trained_model ? "学習済み" : "未学習";
  }

  document.getElementById("btn-train").disabled = status.usable_logs === 0;

  // 学習済み結果エリア
  const resultsEl = document.getElementById("ml-results-section");
  if (status.has_trained_model && status.trained_summary) {
    resultsEl.hidden = false;
    const s = status.trained_summary;
    document.getElementById("ml-interpretation").innerHTML =
      `<strong>モデルの読み取り：</strong> ${escapeHtml(s.interpretation || "")}`;

    renderWeightsChart(s.weights || {});
    const accPct = ((s.top1_accuracy || 0) * 100).toFixed(1);
    document.getElementById("ml-train-caption").textContent =
      `学習イベント数: ${s.num_events_used}件 ｜ Loss: ${s.initial_loss} → ${s.final_loss} ｜ Top-1一致率: ${accPct}%`;

    renderLossChart(status.loss_history || []);

    const compTbody = document.getElementById("ml-comparison-tbody");
    compTbody.innerHTML = (status.comparison_rows || [])
      .map(
        (row) => `
        <tr>
          <td>${escapeHtml(row.pair)}</td>
          <td>${escapeHtml(row.std_top)}</td>
          <td>${escapeHtml(row.learned_top)}</td>
          <td>${row.changed ? '<span class="tag-changed">変化あり ✨</span>' : "同じ"}</td>
        </tr>
      `
      )
      .join("");
  } else {
    resultsEl.hidden = true;
  }

  // 選択ログ一覧とダウンロードボタン
  const downloadBtn = document.getElementById("btn-download-jsonl");
  const emptyEl = document.getElementById("ml-logs-empty");
  const tableWrap = document.getElementById("ml-logs-table-wrap");
  const tbody = document.getElementById("ml-logs-tbody");

  if (status.logs_total > 0) {
    downloadBtn.hidden = false;
    emptyEl.hidden = true;
    tableWrap.hidden = false;
    tbody.innerHTML = (status.recent_logs || [])
      .map(
        (r) => `
        <tr>
          <td>${escapeHtml(r.origin)}</td>
          <td>${escapeHtml(r.destination)}</td>
          <td>${escapeHtml(r.chosen_title)}</td>
          <td>${r.num_candidates}</td>
          <td>${escapeHtml(r.disabled_transfers)}</td>
        </tr>
      `
      )
      .join("");
  } else {
    downloadBtn.hidden = true;
    emptyEl.hidden = false;
    tableWrap.hidden = true;
  }
}

function renderWeightsChart(weightsObj) {
  const wrap = document.getElementById("weights-chart");
  const entries = Object.entries(weightsObj);
  if (entries.length === 0) {
    wrap.innerHTML = "";
    return;
  }
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.1);
  wrap.innerHTML = entries
    .map(([name, val]) => {
      const pct = Math.min(50, (Math.abs(val) / maxAbs) * 50);
      const isNeg = val < 0;
      const style = isNeg
        ? `left:${(50 - pct).toFixed(2)}%;width:${pct.toFixed(2)}%;`
        : `left:50%;width:${pct.toFixed(2)}%;`;
      const cls = isNeg ? "weight-bar-neg" : "weight-bar-pos";
      const valStr = `${val >= 0 ? "+" : ""}${Number(val).toFixed(3)}`;
      return `
        <div class="weight-row">
          <div class="weight-name">${escapeHtml(name)}</div>
          <div class="weight-track">
            <div class="weight-zero"></div>
            <div class="weight-bar ${cls}" style="${style}"></div>
          </div>
          <div class="weight-val">${valStr}</div>
        </div>
      `;
    })
    .join("");
}

function renderLossChart(history) {
  const wrap = document.getElementById("loss-chart");
  if (!history || history.length < 2) {
    wrap.innerHTML = `<p class="chart-caption">「今すぐPyTorchで学習する」を押すとエポックごとの Loss 推移グラフが表示されます。</p>`;
    return;
  }
  const w = 480;
  const h = 170;
  const padL = 42;
  const padR = 14;
  const padT = 14;
  const padB = 26;
  const minY = Math.min(...history);
  const maxY = Math.max(...history);
  const rangeY = Math.max(maxY - minY, 1e-6);

  const points = history
    .map((v, i) => {
      const x = padL + (i / (history.length - 1)) * (w - padL - padR);
      const y = padT + (1 - (v - minY) / rangeY) * (h - padT - padB);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  wrap.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto;display:block;">
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#cbd5e1" stroke-width="1"/>
      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#cbd5e1" stroke-width="1"/>
      <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="${points}"/>
      <text x="${padL - 6}" y="${padT + 8}" font-size="10" fill="#475569" text-anchor="end">${maxY.toFixed(2)}</text>
      <text x="${padL - 6}" y="${h - padB}" font-size="10" fill="#475569" text-anchor="end">${minY.toFixed(2)}</text>
      <text x="${padL}" y="${h - 8}" font-size="10" fill="#475569">Epoch 1</text>
      <text x="${w - padR}" y="${h - 8}" font-size="10" fill="#475569" text-anchor="end">Epoch ${history.length}</text>
    </svg>
  `;
}

// ============================================================================
// イベントリスナー登録 & 初期化
// ============================================================================
function bindEvents() {
  // タブ切り替え
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => {
        const active = b.dataset.tab === target;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", String(active));
      });
      document.getElementById("tab-sim").hidden = target !== "sim";
      document.getElementById("tab-ml").hidden = target !== "ml";
      if (target === "ml") {
        loadMlStatus();
      }
    });
  });

  // クイック体験シナリオボタン
  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = PRESETS[btn.dataset.preset];
      if (!preset) return;
      state.origin = preset.origin;
      state.destination = preset.destination;
      state.priority = preset.priority;
      state.disabledTransfers = new Set(preset.disabledTransfers);
      state.closedSegments = [...preset.closedSegments];
      state.selectedIndex = 0;

      // シミュレータータブへ戻してから検索実行
      document.querySelector('.tab-btn[data-tab="sim"]').click();
      syncAllControls();
      runSearch();
    });
  });

  // 発着駅・優先条件セレクト
  document.getElementById("origin-select").addEventListener("change", (e) => {
    state.origin = e.target.value;
    state.selectedIndex = 0;
    runSearch();
  });
  document.getElementById("destination-select").addEventListener("change", (e) => {
    state.destination = e.target.value;
    state.selectedIndex = 0;
    runSearch();
  });
  document.getElementById("priority-select").addEventListener("change", (e) => {
    state.priority = e.target.value;
    state.selectedIndex = 0;
    runSearch();
  });

  // 発着駅入れ替え
  document.getElementById("swap-btn").addEventListener("click", () => {
    [state.origin, state.destination] = [state.destination, state.origin];
    state.selectedIndex = 0;
    syncAllControls();
    runSearch();
  });

  // 区間閉鎖の追加ドロップダウン
  document.getElementById("segment-add-select").addEventListener("change", (e) => {
    const segId = e.target.value;
    e.target.value = "";
    if (!segId || state.closedSegments.includes(segId)) return;
    state.closedSegments.push(segId);
    state.selectedIndex = 0;
    renderClosedSegmentChips();
    runSearch();
  });

  // 制約を一括解除
  document.getElementById("reset-constraints-btn").addEventListener("click", () => {
    state.disabledTransfers.clear();
    state.closedSegments = [];
    state.selectedIndex = 0;
    syncAllControls();
    runSearch();
  });

  // タブ 2: PyTorch 学習・サンプル投入・リセット
  document.getElementById("btn-train").addEventListener("click", async () => {
    const btn = document.getElementById("btn-train");
    btn.disabled = true;
    try {
      const res = await apiJson("/api/ml/train", {
        method: "POST",
        body: JSON.stringify({ epochs: 250, lr: 0.08, weight_decay: 0.02 }),
      });
      showToast(`🧠 ${res.message}`);
      renderMlStatus(res.ml_status);
      runSearch();
    } catch (err) {
      showToast(err.message, true);
      btn.disabled = false;
    }
  });

  document.getElementById("btn-sample-avoid").addEventListener("click", async () => {
    try {
      const res = await apiJson("/api/ml/sample", {
        method: "POST",
        body: JSON.stringify({ persona: "avoid_transfers" }),
      });
      showToast(`🧪 ${res.message}`);
      renderMlStatus(res.ml_status);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  document.getElementById("btn-sample-short").addEventListener("click", async () => {
    try {
      const res = await apiJson("/api/ml/sample", {
        method: "POST",
        body: JSON.stringify({ persona: "shortest_stops" }),
      });
      showToast(`⚡ ${res.message}`);
      renderMlStatus(res.ml_status);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  document.getElementById("btn-ml-reset").addEventListener("click", async () => {
    try {
      const res = await apiJson("/api/ml/reset", { method: "POST" });
      showToast(`🗑️ ${res.message}`);
      renderMlStatus(res.ml_status);
      runSearch();
    } catch (err) {
      showToast(err.message, true);
    }
  });
}

async function initApp() {
  bindEvents();
  try {
    const meta = await apiJson("/api/meta");
    state.meta.stations = meta.stations;
    state.meta.transfers = meta.transfers;
    state.meta.segments = meta.segments;
    state.meta.segmentMap = new Map(meta.segments.map((s) => [s.id, s]));

    // 発着駅ドロップダウンの構築
    const originSel = document.getElementById("origin-select");
    const destSel = document.getElementById("destination-select");
    const stationOptionsHtml = meta.stations
      .map((st) => `<option value="${escapeHtml(st.name)}">${escapeHtml(st.label)}</option>`)
      .join("");
    originSel.innerHTML = stationOptionsHtml;
    destSel.innerHTML = stationOptionsHtml;

    // 閉鎖区間ドロップダウンの構築
    const segSel = document.getElementById("segment-add-select");
    segSel.innerHTML =
      `<option value="">閉鎖したい区間を選択して追加...</option>` +
      meta.segments
        .map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.label)}</option>`)
        .join("");

    syncPriorityOption(meta.has_trained_model);
    syncAllControls();

    // 初回描画（経路探索 + ML ステータス取得を並列実行）
    await Promise.all([runSearch(), loadMlStatus()]);
  } catch (err) {
    showToast(`初期化エラー: ${err.message}`, true);
  }
}

document.addEventListener("DOMContentLoaded", initApp);
