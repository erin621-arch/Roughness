import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib
import os 
import japanize_matplotlib

output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

# ====== パラメータ ======
mesh_length = 1.0e-5
nx = 2000
nz = 4000
f_pitch = 1.25e-3
f_depth = 0.20e-3
step_size = 1
mn_p = int(round(f_pitch / mesh_length))  # 125
mn_d = int(round(f_depth / mesh_length))  # 20

sz = int(nz / 2)  # 2000（探触子中心）

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
corner_z = []
corner_x = []
for i in range(len(z_surface) - 1):
    if depth_vals[i+1] < depth_vals[i]:
        corner_z.append(int(z_surface[i+1]))
        corner_x.append(int(x_surface[i]))
corner_z = np.array(corner_z)
corner_x = np.array(corner_x)

# ====== 描画 ======
fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle("くさび表面のT3 の計測点の位置確認\n"
             f"（pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.2f} mm ) ",fontsize=11)

z_lo = z_rec_start - 20
z_hi = z_rec_end   + 20
x_depth_show = mn_d + 15
ann_space    = 20

ax.set_xlim(z_lo, z_hi)
ax.set_ylim(nx + ann_space, nx - x_depth_show)  # 大きい値が下（inverted y）
ax.set_xlabel("z 方向 [mm]",fontsize=11, labelpad=6)
ax.set_ylabel("x 方向 [mm]", fontsize=11, labelpad=6)
ax.set_aspect('equal')

# 材料ブロック（背景）
ax.add_patch(mpatches.Rectangle(
    (z_lo, nx - x_depth_show), z_hi - z_lo, x_depth_show,
    facecolor="lightyellow", edgecolor="black", lw=2.0, zorder=1
))

# くさびプロファイル（空洞）
z_arr_full = np.arange(z_lo, z_hi + 1)
depth_arr  = np.array([kusabi_depth(z) for z in z_arr_full], dtype=float)
surf_arr   = nx - depth_arr
ax.fill_between(z_arr_full, surf_arr, nx, color="white", step="post", zorder=2)
ax.step(z_arr_full, surf_arr, color="black", lw=1.5, where="post", zorder=4)

# 表面ライン
ax.axhline(y=nx, color="black", lw=2.0, zorder=5)

# 探触子中心ライン
ax.axvline(x=sz, color="royalblue", lw=1.2, ls="--", alpha=0.9, zorder=6,
           label=f"探触子中心 z={sz*mesh_length*1e3:.2f} mm")

# 外側の凸角（計測点）
ax.scatter(corner_z, corner_x,
           s=50, color="red", marker="o", zorder=7,
           label=f"計測点")

# 垂直壁の説明
ax.annotate("", xy=(z_rec_start, nx - mn_d), xytext=(z_rec_start, nx),
            arrowprops=dict(arrowstyle="<->", color="sienna", lw=1.5), zorder=6)
ax.text(z_rec_start - 3, nx - mn_d / 2,
        f"depth\n{f_depth*1e3:.2f}mm",
        ha="right", va="center", fontsize=15, color="sienna")

# ピッチ幅の矢印
y_ann = nx + 12
ax.annotate("", xy=(z_rec_end, y_ann), xytext=(z_rec_start, y_ann),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5), zorder=7)
ax.text((z_rec_start + z_rec_end) / 2, y_ann + 1.5,
        f"pitch: {f_pitch*1e3:.2f} mm",
        ha="center", va="top", fontsize=15, color="purple")

# 理想三角形の斜辺（点線）
ax.plot([z_rec_start, z_rec_end], [nx - mn_d, nx],
        color="blue", lw=1.5, ls="--", zorder=3, label="理想斜面")

# 凡例
ax.legend(loc="upper right", fontsize=15)

# 軸ラベル（mm 表記）
ax.set_yticks([nx - x_depth_show, nx - mn_d, nx])
ax.set_yticklabels([f"{(nx-x_depth_show)*mesh_length*1e3:.2f}\n",
                    f"{(nx-mn_d)*mesh_length*1e3:.2f}\n(最大深さ)",
                    f"{nx*mesh_length*1e3:.2f}\n(表面)"], fontsize=15)
xticks = np.arange(z_rec_start - 20, z_rec_end + 30, 25)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{v*mesh_length*1e3:.2f}" for v in xticks], fontsize=15)

# 1メッシュ単位のグリッド線（直接描画）
for _z in range(z_lo, z_hi + 1):
    ax.axvline(x=_z, color='lightgray', linewidth=0.3, zorder=3)
for _x in range(nx - x_depth_show, nx + ann_space + 1):
    ax.axhline(y=_x, color='lightgray', linewidth=0.3, zorder=3)

plt.tight_layout(rect=[0, 0, 1, 0.92])

fig_name = os.path.join(
    output_dir,
    f"kusabi_T3_map_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)

plt.savefig(fig_name, dpi=150, bbox_inches="tight")
print(f"saved: {fig_name}")
plt.show()
