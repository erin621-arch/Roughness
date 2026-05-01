"""
ushape_surface_sim_center_v2.py の結果プロットスクリプト

2パネル構成:
  [上] T1 連結マップ: 直線部（右壁 z=2025, d=0..7）+ アーク部（コーナー点 Pt1-Pt7）
         y軸を x インデックスで統一し、測定なし格子は灰色表示
  [下] T3_gap: ギャップ連続記録（従来通り）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import japanize_matplotlib
import os

# ====== ファイル指定 ======
input_dir  = r"C:/Users/cs16/Roughness/project4/tmp_output"
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"
f_pitch = 2.00e-3
f_depth = 0.20e-3

npz_path = os.path.join(
    input_dir,
    f"ushape_surface_map_center_v2_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

# ====== データ読み込み ======
d = np.load(npz_path)

T1_straight  = d['T1_straight']              # (mn_straight, t_rec_len)
T1_corners_r = d['T1_corners_r']             # (n_arc_pts,   t_rec_len)
T3_gap       = d['T3_gap']                   # (gap_width,   t_rec_len)

straight_x   = d['straight_x'].astype(int)  # [1992..1999]
straight_z   = int(d['straight_z'])          # 2025
mn_straight  = int(d['mn_straight'])         # 8
corner_x_r   = d['corner_x_r'].astype(int)
corner_z_r   = d['corner_z_r'].astype(int)
gap_z_start  = int(d['gap_z_start'])
gap_z_end    = int(d['gap_z_end'])
t_rec_start  = int(d['t_rec_start'])
t_rec_len    = int(d['t_rec_len'])
dt           = float(d['dt'])
mesh_length  = float(d['mesh_length'])
f_pitch      = float(d['f_pitch'])
f_depth      = float(d['f_depth'])
mn_d         = int(d['mn_d'])
mn_r         = int(d['mn_r'])
mn_straight  = int(d['mn_straight'])

nx     = straight_x[-1] + 1   # 2000
n_arc  = T1_corners_r.shape[0]

# ====== 軸の構築 ======
t_axis     = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6  # [µs]
gap_z_axis = np.arange(gap_z_start, gap_z_end) * mesh_length * 1e3  # [mm]

# ====== 連結 T1 マップの構築 ======
# x = x_min(Pt1=1980) 〜 x_max(直線部入口=1999)  →  20 行
x_min   = int(corner_x_r.min())   # 1980
x_max   = int(straight_x[-1])     # 1999
n_rows  = x_max - x_min + 1       # 20

# NaN で初期化（測定なし格子 = 灰色表示）
combined_T1 = np.full((n_rows, t_rec_len), np.nan)

# アーク部コーナー点を埋める（Pt1-Pt7）
for i in range(n_arc):
    row = corner_x_r[i] - x_min
    combined_T1[row, :] = T1_corners_r[i, :]

# 直線部右壁を埋める（x=1992..1999）
for i, xi in enumerate(straight_x):
    row = xi - x_min
    combined_T1[row, :] = T1_straight[i, :]

# 浅い方（x=1999）を上に表示 → 行方向を反転
combined_T1_disp = combined_T1[::-1, :]   # row0 = x=1999, row19 = x=1980

# 欠損セルをマスク
masked_T1 = np.ma.masked_invalid(combined_T1_disp)
cmap_T1 = plt.cm.bwr.copy()
cmap_T1.set_bad(color='lightgray')

vmax_T1 = np.nanpercentile(np.abs(combined_T1), 99)
vmax_T3 = np.percentile(np.abs(T3_gap), 99)

# y 軸ラベル用: 表示順は x=1999(top) → x=1980(bottom)
x_disp = np.arange(x_max, x_min - 1, -1)   # [1999, 1998, ..., 1980]
d_disp = nx - 1 - x_disp                    # [0, 1, ..., 19]

# ====== 描画 ======
fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(
    f'Ushape 右壁 T1 連結マップ  (pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.3f} mm)\n'
    f'直線部 (z={straight_z}, d=0〜{mn_straight-1}) + アーク部コーナー (Pt1-Pt{n_arc})\n'
    f'灰色 = 測定なし格子',
    fontsize=11
)

# ----------------------------------------------------------------
# [上] 連結 T1 マップ
# ----------------------------------------------------------------
ax = axes[0]

im0 = ax.imshow(
    masked_T1,
    aspect='auto',
    cmap=cmap_T1,
    vmin=-vmax_T1, vmax=vmax_T1,
    extent=[t_axis[0], t_axis[-1], x_min - 0.5, x_max + 0.5],
    interpolation='nearest',
    origin='upper'
)
plt.colorbar(im0, ax=ax, label='T1 [Pa]')

ax.set_ylabel('x インデックス（深さ方向）', fontsize=10)
ax.set_title('T1 右壁 連結マップ（上 = 浅い / 下 = 深い）', fontsize=10)

# y 軸目盛り: 各測定点
measured_xs = sorted(set(corner_x_r.tolist() + straight_x.tolist()), reverse=True)
ax.set_yticks(measured_xs)
labels = []
for xi in measured_xs:
    di = nx - 1 - xi
    if xi in straight_x:
        labels.append(f'x={xi}  d={di}  直線部')
    else:
        pt_idx = np.where(corner_x_r == xi)[0][0] + 1
        labels.append(f'x={xi}  d={di}  Pt{pt_idx}')
ax.set_yticklabels(labels, fontsize=7)

# 境界線
x_straight_bot = int(straight_x[0])   # 1992（直線部最深行）
x_arc_top      = x_straight_bot - 1   # 1991（アーク部上端）

ax.axhline(y=x_straight_bot - 0.5, color='darkorange', linestyle='--',
           linewidth=1.5, label=f'直線部/アーク境界 (x={x_straight_bot}|{x_arc_top})')
ax.axhline(y=x_max + 0.5 - 1, color='purple', linestyle=':',
           linewidth=1.2, label=f'直線部入口 d=0 (x={x_max})')
ax.legend(fontsize=8, loc='upper right')
ax.invert_yaxis()

# ----------------------------------------------------------------
# [下] ギャップ T3 連続記録
# ----------------------------------------------------------------
ax = axes[1]

im1 = ax.imshow(
    T3_gap,
    aspect='auto',
    cmap='bwr',
    vmin=-vmax_T3, vmax=vmax_T3,
    extent=[t_axis[0], t_axis[-1], gap_z_axis[-1], gap_z_axis[0]],
    interpolation='bilinear'
)
plt.colorbar(im1, ax=ax, label='T3 [Pa]')
ax.set_ylabel('z [mm]', fontsize=10)
ax.set_title('T3 ギャップ連続記録', fontsize=10)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]', fontsize=10)

plt.tight_layout()

fig_path = os.path.join(
    output_dir,
    f"ushape_T1_v2_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"保存: {fig_path}")
plt.show()
