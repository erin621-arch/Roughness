
---

## フォルダ構成

### データ関連

#### `Experiment_Data/`
実機計測データの格納フォルダ。
`[Shape]/[pitch]_[depth]/` の構成でサブフォルダが作られており、形状（Kukei・Kusabi・Sankaku・Smooth・Ushape）×ピッチ・深さの組み合わせごとに計測波形を管理する。

#### `Simulation_Data/`
FDTD シミュレーション結果の格納フォルダ。
`[Shape]/[shape]_cupy_pitch[pitch]_depth[depth].csv` の命名規則でデータを保存。
`step~` サフィックスの付いたファイルはステップサイズ変更バリアント。
対象形状：Kukei・Kusabi・Sankaku・Smooth・Ushape

#### `View_Data/`
可視化用に整理済みのデータ群。
`Kusabi/`・`Sankaku/`・`Ushape/`・`V_plot_datas/` のサブフォルダを含む。

#### `tmp_output/`
シミュレーション中間出力・デバッグ用の一時ファイル置き場。

---

### スクリプト関連

#### `Plot_programs/`
各種プロット・可視化スクリプト群。

| ファイル | 説明 |
|---|---|
| `Experiment_amp_plot.py` | 実験データの振幅スペクトルプロット |
| `Simulation_amp_plot.py` | シミュレーションデータの振幅スペクトルプロット |
| `kusabi_T1_T3_power_plot.py` | くさび形状の応力成分（T1・T3）パワープロット |
| `ushape_T1_T3_power_plot.py` | U字形状の応力成分（T1・T3）パワープロット |
| `talewaveplot_depth.py` | 深さ別波形比較プロット |
| `talewaveplot_stepsize.py` | ステップサイズ別波形比較プロット |
| `V_plot_data_1.py` ～ `V_plot_data_6.py` | V 字プロット用データ処理スクリプト群 |

#### `past_files/`
旧バージョンのスクリプト保管フォルダ。
`sankaku_ani.py`・`sankaku_sim.py`・`kusabi_and_ushape_plot.py`・`kukei_one_plot.py`・`Ushape_one_plot.py` などを含む。

---

## 主要ソースファイル（ルート直下）

| ファイル | 説明 |
|---|---|
| `ushape_sim.py` | U字形状の FDTD シミュレーション実行スクリプト |
| `ushape_ani.py` | U字形状シミュレーション結果のアニメーション生成 |
| `kusabi_sim.py` | くさび形状の FDTD シミュレーション実行スクリプト |
| `kusabi_ani.py` | くさび形状シミュレーション結果のアニメーション生成 |
| `FFT_one.py` | 単一データの FFT 解析スクリプト |
| `FFT_multi.py` | 複数データの FFT 一括解析スクリプト |
| `npz_player.py` | `.npz` 形式データの読み込み・表示 |
| `sourse_2.py` | 信号処理・データ読み込みのコアユーティリティ |
| `sourse_new_2.py` | `sourse_2.py` の改良版 |
| `sourse_0211.py` | 2月11日時点のチェックポイント版 |

---

## 命名規則

- **パラメータのコード値**：整数値は実際の値 × 10⁵（例：`12500000` = `125` mm 相当）
- **Experiment_Data**：`[Shape]/[pitch]_[depth]/`
- **Simulation_Data**：`[Shape]/[shape]_cupy_pitch[pitch]_depth[depth].csv`
- **step~ ファイル**：ステップサイズを変更したシミュレーションバリアント

---

## 対象形状一覧

| フォルダ名 | 形状 |
|---|---|
| `Kukei` | 矩形（くけい）欠陥 |
| `Kusabi` | くさび形欠陥 |
| `Sankaku` | 三角形欠陥 |
| `Smooth` | スムーズ欠陥 |
| `Ushape` | U字形欠陥 |
