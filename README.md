# 🚇 東京メトロ 乗換シミュレーター（MVP: 銀座線＋丸ノ内線）

東京メトロの実際の路線網（銀座線・丸ノ内線＋方南町支線）をグラフ化し、通常の乗換検索に加えて **「この駅で乗り換えられなかったら？」「この区間が止まったら？」** を路線図上で試せる Streamlit ＋ PyTorch アプリです。

## 主な機能

1. **駅×路線ノードのグラフ構造** (`data/tokyo_metro_mvp.json`, `src/metro_graph.py`)
   - `銀座・銀座線 (G-09)` と `銀座・丸ノ内線 (M-16)` を別ノードとして表現。
   - 隣駅への「乗車エッジ」と路線を変える「乗換エッジ」を分離しているため、**異駅名の公式乗換（溜池山王 ↔ 国会議事堂前）** も自然に表現できます。
   - 丸ノ内線の方南町支線（中野坂上分岐）にも対応しています。
   - JSONに路線・駅・乗換を追加するだけで、東京メトロ全9路線へそのまま拡張できます。
2. **ダイクストラ法＋多様候補探索** (`src/routing.py`)
   - **乗換が少ない順** / **乗車区間数が少ない順** の2基準で探索。
   - **乗換無効化**：指定駅での路線乗換だけを除外し、通過列車はそのまま利用可能。
   - **区間閉鎖モード**：指定した隣駅間（双方向）を閉鎖して迂回経路をシミュレーション。
   - 変更前後の第1候補・除外された候補を並べて比較表示。
3. **PyTorch 好み学習 & Colab 連携** (`src/preference_model.py`, `notebooks/tokyo_metro_preference_colab.ipynb`)
   - 候補横の **「👍 私ならこの経路を選ぶ」** で選択ログを `data/user_choices.jsonl` に記録。
   - PyTorchの線形効用モデル（Multinomial Logit / Listwise Cross-Entropy）を初期重み `0` から学習し、**「乗換1回を減らすためなら何駅分の遠回りを許容するか」** を推定して候補を再ランキングします。
   - 選択ログはワンクリックでダウンロードし、Google Colabノートブックで学習・モデル比較（Linear vs MLP）・並び順変化の分析を行えます。

## セットアップ & 起動方法

```bash
cd /Users/apple/.gemini/antigravity/scratch/tokyo-metro-simulator

# 仮想環境を作成して依存パッケージをインストール
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Streamlit アプリを起動
.venv/bin/streamlit run app.py
```

## テストの実行

```bash
.venv/bin/pytest -v
```

## 東京メトロ全9路線への拡張手順

`data/tokyo_metro_mvp.json` に以下の形式で残り7路線（日比谷線 `H`、東西線 `T`、千代田線 `C`、有楽町線 `Y`、半蔵門線 `Z`、南北線 `N`、副都心線 `F`）を追加するだけで、探索エンジン・路線図レンダラー・PyTorch学習モデルがそのまま動作します。

- `lines`: 路線ID・名称・公式カラー
- `nodes`: 各駅×路線ノード（駅ナンバリング・座標）
- `ride_segments`: 隣駅間の接続と方面名
- `transfers`: 公式乗換ペア（淡路町 ↔ 新御茶ノ水、赤坂見附 ↔ 永田町などの異駅名乗換は `is_diff_name: true`）
