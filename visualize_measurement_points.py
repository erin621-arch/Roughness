import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans JP'

# ====== パラメータ ======
mesh_length = 1.0e-5
nx = 2000
nz = 4000
f_pitch = 1.25e-3
f_depth = 0.20e-3
step_size = 1
mn_p = int(round(f_pitch / mesh_length))  # 125
mn_d = int(round(f_depth / mesh_length))  # 20

sz = int(nz / 2)  # 2000

# ====== 計測ピッチの設定 ======
pitch_i     = sz // mn_p        # 16
z_rec_start = pitch_i * mn_p    # 2000
z_rec_end   = (pitch_i + 1) * mn_p  # 2125

z_surface = np.arange(z_rec_start, z_rec_end + 1)  # 2000〜2125

def kusabi_depth(z):
    local_z = z - (z // mn_p) * mn_p
    ideal   = mn_d * (1.0 - local_z / mn_p)
    return (int(ideal) // step_size) * step_size

x_surface = np.array([nx - kusabi_depth(z) for z in z_surface])
depth_vals = np.array([kusabi_depth(z) for z in z_surface])

# ====== 外側の凸角の座標を計算 ======
# 段の末端は「深さが次のzで減少する時の z+1 の位置（段が上がる始点）」であり、
# x座標は前の段（深い方）の高さを使う。
corner_z = []
corner_x = []
for i in range(len(z_surface) - 1):
    if depth_vals[i+1] < depth_vals[i]:
        corner_z.append(int(z_surface[i+1]))   # 段が上がる z 位置
        corner_x.append(int(x_surface[i]))     # 前の段（深い側）の x
corner_z = np.array(corner_z)
corner_x = np.array(corner_x)

# ====== 底面点（理想くさびの先端）======
# z_rec_end=2125, x=nx=2000（底面の平坦部レベル）
apex_z = z_rec_end  # 2125
apex_x = nx         # 2000

# ====== 描画 ======
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"width_ratios": [1, 1.4]})
fig.suptitle("くさび表面の T1 計測点の位置確認\n"
             f"（pitch=1.25mm, depth=0.20mm, ピッチ i={pitch_i}, z={z_rec_start}〜{z_rec_end}）",
             fontsize=12)

# ========== 左パネル：全体俯瞰 ==========
ax = axes[0]
margin_z = 300
ax.set_xlim(-margin_z, nz + margin_z)
ax.set_ylim(nx + 60, 0)
ax.set_xlabel("z 方向 [mesh]", labelpad=6)
ax.set_ylabel("x 方向（深さ） [mesh]", labelpad=6)
ax.set_title("全体俯瞰（計測ピッチを強調）")
ax.set_facecolor("white")

# 金属ブロック
ax.add_patch(mpatches.Rectangle(
    (0, 0), nz, nx,
    facecolor="lightyellow", edgecolor="none", zorder=0
))
ax.axvline(x=0,  color="black", lw=2.0, zorder=5)
ax.axvline(x=nz, color="black", lw=2.0, zorder=5)
ax.hlines(y=nx, xmin=0, xmax=nz, color="black", lw=2.0, zorder=5)
ax.hlines(y=0,  xmin=0, xmax=nz, color="dimgray", lw=1.0, ls="--", zorder=4)

# 全ピッチのくさびを薄く描画
num_f = int(np.ceil(nz / mn_p))
from matplotlib.patches import Polygon
for i in range(num_f):
    z_s = i * mn_p
    z_e = min((i + 1) * mn_p, nz)
    verts = np.array([[z_s, nx], [z_s, nx - mn_d], [z_e, nx]])
    ax.add_patch(Polygon(verts, closed=True,
                         facecolor="lightgray", edgecolor="gray", lw=0.3, alpha=0.5, zorder=2))

# 計測ピッチを強調
verts_rec = np.array([[z_rec_start, nx], [z_rec_start, nx - mn_d], [z_rec_end, nx]])
ax.add_patch(Polygon(verts_rec, closed=True,
                     facecolor="tomato", edgecolor="crimson", lw=1.5, alpha=0.7, zorder=3))

# 計測点：外側の凸角 + 底面点
all_meas_z = np.append(corner_z, apex_z)
all_meas_x = np.append(corner_x, apex_x)
ax.scatter(all_meas_z, all_meas_x,
           s=14, color="red", zorder=6,
           label=f"計測点（{len(all_meas_z)}点）")

# 探触子バー
probe_d = 0.007
sz_l = sz - int(probe_d / mesh_length / 2)
sz_r = sz + int(probe_d / mesh_length / 2)
probe_h = int(nx * 0.012)
ax.add_patch(mpatches.Rectangle(
    (sz_l, 0), sz_r - sz_l, probe_h,
    facecolor="royalblue", edgecolor="black", lw=0.8, alpha=0.9, zorder=4
))
ax.axvline(x=sz, color="royalblue", lw=0.9, ls="--", alpha=0.7)

ax.legend(loc="lower right", fontsize=8)
ax.text(-margin_z * 0.5, nx / 2, "空気", ha="center", va="center",
        fontsize=9, color="gray", rotation=90, style="italic")
ax.text(nz + margin_z * 0.5, nx / 2, "空気", ha="center", va="center",
        fontsize=9, color="gray", rotation=90, style="italic")

# ========== 右パネル：計測ピッチの拡大図 ==========
ax2 = axes[1]

z_lo = z_rec_start - 20
z_hi = z_rec_end   + 20
x_depth_show = mn_d + 15
ann_space    = 20

ax2.set_xlim(z_lo, z_hi)
ax2.set_ylim(nx + ann_space, nx - x_depth_show)  # 大きい値が下（inverted y）
ax2.set_xlabel("z 方向 [mesh]  (1 mesh = 10 µm)", labelpad=6)
ax2.set_ylabel("x 方向 [mesh]", labelpad=6)
ax2.set_title(f"拡大図：ピッチ i={pitch_i} の計測点\n"
              f"（赤丸 = T1 計測点, 計{len(corner_z) + 1}点）")

# 材料ブロック（背景）
ax2.add_patch(mpatches.Rectangle(
    (z_lo, nx - x_depth_show), z_hi - z_lo, x_depth_show,
    facecolor="lightyellow", edgecolor="black", lw=2.0, zorder=1
))

# くさびプロファイル（空洞）
z_arr_full = np.arange(z_lo, z_hi + 1)
depth_arr  = np.array([kusabi_depth(z) for z in z_arr_full], dtype=float)
surf_arr   = nx - depth_arr
ax2.fill_between(z_arr_full, surf_arr, nx, color="white", step="post", zorder=2)
ax2.step(z_arr_full, surf_arr, color="black", lw=1.5, where="post", zorder=4)

# 表面ライン
ax2.axhline(y=nx, color="black", lw=2.0, zorder=5)

# その他の表面点（凸角・底面点以外）
# 凸角の z 集合
corner_z_set = set(corner_z.tolist())
is_normal = np.array([
    (int(z) not in corner_z_set) and (int(z) != apex_z)
    for z in z_surface
], dtype=bool)
ax2.scatter(z_surface[is_normal], x_surface[is_normal],
            s=12, color="lightcoral", alpha=0.5, zorder=5,
            label=f"その他の表面点（{is_normal.sum()}点）")

# 外側の凸角 + 底面点（同じ赤丸で表示）
ax2.scatter(np.append(corner_z, apex_z), np.append(corner_x, apex_x),
            s=50, color="red", marker="o", zorder=7,
            label=f"計測点（凸角+底面点, 計{len(corner_z) + 1}点）")

# 垂直壁の説明
ax2.annotate("", xy=(z_rec_start, nx - mn_d), xytext=(z_rec_start, nx),
             arrowprops=dict(arrowstyle="<->", color="sienna", lw=1.5), zorder=6)
ax2.text(z_rec_start - 3, nx - mn_d / 2,
         f"depth\n{f_depth*1e3:.2f}mm",
         ha="right", va="center", fontsize=8, color="sienna")

# ピッチ幅の矢印（注釈スペース）
y_ann = nx + 12
ax2.annotate("", xy=(z_rec_end, y_ann), xytext=(z_rec_start, y_ann),
             arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5), zorder=7)
ax2.text((z_rec_start + z_rec_end) / 2, y_ann + 1.5,
         f"pitch: {f_pitch*1e3:.2f} mm ({mn_p} mesh)",
         ha="center", va="top", fontsize=9, color="purple")

# 理想三角形の斜辺（点線）
ax2.plot([z_rec_start, z_rec_end], [nx - mn_d, nx],
         color="blue", lw=1.5, ls="--", zorder=3, label="理想斜面（近似前）")

# 凡例
ax2.legend(loc="lower left", fontsize=8)

# 軸ラベル
ax2.set_yticks([nx - x_depth_show, nx - mn_d, nx])
ax2.set_yticklabels([f"{nx-x_depth_show}\n(材料内)",
                     f"{nx-mn_d}\n(最大深さ)",
                     f"{nx}\n(表面)"], fontsize=7.5)
xticks = np.arange(z_rec_start - 20, z_rec_end + 30, 25)
ax2.set_xticks(xticks)
ax2.set_xticklabels([f"{v}\n({v*mesh_length*1e3:.2f}mm)" for v in xticks], fontsize=7)

plt.tight_layout()
plt.savefig("measurement_points_kusabi.png", dpi=150, bbox_inches="tight")
print("saved: measurement_points_kusabi.png")
plt.show()
