
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

---

## 主要ソースファイル（ルート直下）

| ファイル | 説明 |
|---|---|
| `ushape_sim.py` | U字形状の FDTD シミュレーション実行スクリプト |
| `kusabi_sim.py` | くさび形状の FDTD シミュレーション実行スクリプト |
| `sourse_2.py` | 信号処理・データ読み込みのコアユーティリティ |
| `sourse_new_2.py` | `sourse_2.py` の改良版 |
| `sourse_0211.py` | 2月11日時点のチェックポイント版 |

---

## 命名規則

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
| `Smooth` | 平滑面 |
| `Ushape` | U字形欠陥 |
