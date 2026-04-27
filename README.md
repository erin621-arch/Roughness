
---

# project4

FDTD法による弾性波シミュレーションと実験データの比較・解析プログラム群。
傷形状（くさび・U字・矩形・三角・スムーズ）ごとにシミュレーションと実験を対応させる。

---

## フォルダ構成

### データ関連

#### `Experiment_Data/`
実機計測データの格納フォルダ。
`[Shape]/[pitch]_[depth]/` の構成でサブフォルダが作られており、形状×ピッチ・深さの組み合わせごとに計測波形を管理する。

#### `Simulation_Data/`
FDTD シミュレーション結果の格納フォルダ。
`[Shape]/[shape]_cupy_pitch[pitch]_depth[depth].csv` の命名規則でデータを保存。
`step~` サフィックスの付いたファイルはステップサイズ変更バリアント。

#### `View_Data/`
可視化用に整理済みのデータ群。
`Kusabi/`・`Sankaku/`・`Ushape/`・`V_plot_datas/` のサブフォルダを含む。

#### `tmp_output/`
一時ファイル置き場。

---

### スクリプト関連

#### `202604_kusabi/`
くさび形状に関するシミュレーション・解析スクリプトを集約したフォルダ（2026-04以降）。

| ファイル | 説明 |
|---|---|
| `kusabi_surface_sim_edge.py` | T1/T3 計測・端励起シミュレーション |
| `kusabi_surface_sim_center.py` | T1/T3 計測・ピッチ中心励起シミュレーション |
| `kusabi_surface_sim_sigma.py` | T1/T3/T5 計測・sigma 合力計算（T5 統合版） |
| `kusabi_map_points.py` | 計測点位置確認図の生成 |
| `plot_kusabi_T1_T3_surface_map.py` | T1/T3 パワーマッププロット |
| `plot_kusabi_T1_T_sigma_surface_map.py` | T1/σ パワーマッププロット |
| `plot_kusabi_T135_slope_check.py` | T1/T3/T5 斜面チェック用プロット |
| `kusabi_sigma_summary.ipynb` | くさびシミュレーション結果まとめノートブック |

#### `202604_ushape/`
U字形状に関するシミュレーション・解析スクリプトを集約したフォルダ（2026-04以降）。

| ファイル | 説明 |
|---|---|
| `ushape_surface_sim_center.py` | U字形状シミュレーション |
| `Ushape_map_points.py` | 計測点位置確認図の生成 |
| `plot_ushape_T1_T3_surface_map.py` | T1/T3 パワーマッププロット |

#### `Simulation_flies/`
シミュレーション実行スクリプト（旧版）。

| ファイル | 説明 |
|---|---|
| `ushape_sim.py` | U字形状の FDTD シミュレーション実行 |
| `kusabi_sim.py` | くさび形状の FDTD シミュレーション実行 |

#### `Sourse/`
信号処理・データ読み込みのコアユーティリティ。

| ファイル | 説明 |
|---|---|
| `sourse_new_2.py` | 現行版コアユーティリティ |
| `sourse_2.py` | 旧版 |
| `sourse_0211.py` | 2月11日時点のチェックポイント版 |

#### `1_past_files/`
旧バージョンスクリプトの保管フォルダ。

---

## 命名規則

- **パラメータのコード値**：整数値は実際の値 × 10⁵（例：pitch=25000 → 0.25 mm）
- **Experiment_Data**：`[Shape]/[pitch]_[depth]/`
- **Simulation_Data**：`[Shape]/[shape]_cupy_pitch[pitch]_depth[depth].csv`
- **step~ ファイル**：ステップサイズを変更したシミュレーションバリアント

---

## 対象形状一覧

| フォルダ名 | 形状 |
|---|---|
| `Kukei` | 矩形欠陥 |
| `Kusabi` | くさび形欠陥 |
| `Sankaku` | 三角形欠陥 |
| `Smooth` | 平滑面 |
| `Ushape` | U字形欠陥 |

---

## 軸定義

- x 軸：縦方向（下方向が増加）
- z 軸：横方向
