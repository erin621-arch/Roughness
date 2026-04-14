import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans JP'

# ====== パラメータ（kusabi_sim.py から） ======
mesh_length = 1.0e-5
x_length = 0.02
z_length = 0.04
nx = int(round(x_length / mesh_length))  # 2000
nz = int(round(z_length / mesh_length))  # 4000

f_pitch = 1.25e-3
f_depth = 0.20e-3
step_size = 1

mn_p = int(round(f_pitch / mesh_length))  # 125
mn_d = int(round(f_depth / mesh_length))  # 20

num_f = int(np.ceil(nz / mn_p))  # 32 ピッチ

probe_d = 0.007  # 7 mm
sz   = int(nz / 2)                          # 2000 (領域中央)
sz_l = sz - int(probe_d / mesh_length / 2)  # 1650
sz_r = sz + int(probe_d / mesh_length / 2)  # 2350

# ====== くさび深さプロファイル計算 ======
def kusabi_depth(z):
    """z 位置における深さ（メッシュ数）を返す"""
    local_z = z % mn_p
    ideal = mn_d * (1.0 - local_z / mn_p)
    return (int(ideal) // step_size) * step_size

# ====== 描画 ======
fig, axes = plt.subplots(1, 2, figsize=(15, 9),
                         gridspec_kw={"width_ratios": [1, 1.7]})
fig.suptitle(
    "探触子と傷の位置関係（くさび形）\n"
    f"pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.2f} mm, probe_d={probe_d*1e3:.0f} mm",
    fontsize=13
)

# ===========================
# 左パネル：全体俯瞰
# ===========================
ax = axes[0]
margin_z = 400  # z 方向の空気層の表示幅 [mesh]
ax.set_xlim(-margin_z, nz + margin_z)
ax.set_ylim(nx + 60, 0)   # x=0 が上、x=nx が下
ax.set_xlabel("z 方向 [mesh]", labelpad=6)
ax.set_ylabel("x 方向（深さ） [mesh]", labelpad=6)
ax.set_title("全体俯瞰\n（上: 探触子面 x=0、下: 背面 x=nx）")

# 背景を白（空気）に設定
ax.set_facecolor("white")
for sp in ax.spines.values():
    sp.set_edgecolor("black")
    sp.set_linewidth(1.0)

# 金属ブロック（黄色 + 境界なし）
ax.add_patch(mpatches.Rectangle(
    (0, 0), nz, nx,
    facecolor="lightyellow", edgecolor="none", zorder=0
))

# 横方向境界ライン（z = 0 と z = nz）
ax.axvline(x=0,  color="black", lw=2.0, zorder=6)
ax.axvline(x=nz, color="black", lw=2.0, zorder=6)

# 横方向の面ラベル
ax.text(-margin_z * 0.5, nx / 2, "空気",
        ha="center", va="center", fontsize=10, color="gray",
        rotation=90, style="italic")
ax.text(nz + margin_z * 0.5, nx / 2, "空気",
        ha="center", va="center", fontsize=10, color="gray",
        rotation=90, style="italic")

# 深さ方向ライン（x = nx：背面、x = 0：探触子面）
ax.hlines(y=nx, xmin=0, xmax=nz, color="black", lw=2.5, zorder=6)
ax.text(nz * 0.02, nx + 8, "背面（傷面）  x = nx = 2000",
        ha="left", va="top", fontsize=8, color="black")
ax.hlines(y=0, xmin=0, xmax=nz, color="dimgray", lw=1.5, ls="--", zorder=6)
ax.text(nz * 0.02, 8, "探触子面  x = 0",
        ha="left", va="top", fontsize=8, color="dimgray")

# くさび形溝（三角形ポリゴン）
for i in range(num_f):
    z_s = i * mn_p
    z_e = min((i + 1) * mn_p, nz)
    verts = np.array([
        [z_s, nx],        # 左上（表面・最深部入口）
        [z_s, nx - mn_d], # 左下（最深部）
        [z_e, nx],        # 右上（表面・深さ0）
    ])
    ax.add_patch(Polygon(verts, closed=True,
                         facecolor="white", edgecolor="black", lw=0.4, zorder=3))

# 探触子バー（x=0 面）
probe_h_left = int(nx * 0.012)
ax.add_patch(mpatches.Rectangle(
    (sz_l, 0), sz_r - sz_l, probe_h_left,
    facecolor="royalblue", edgecolor="black", lw=0.8, alpha=0.9, zorder=4
))
ax.axvline(x=sz, color="royalblue", lw=0.9, ls="--", alpha=0.7, zorder=3)

legend_elements = [
    mpatches.Patch(facecolor="lightyellow", edgecolor="black", lw=1.5, label="金属（鋼材）"),
    mpatches.Patch(facecolor="white",       edgecolor="black", lw=0.5, label="くさび形溝"),
    mpatches.Patch(facecolor="royalblue",   edgecolor="black", lw=0.8,
                   label=f"探触子（sz±{int(probe_d/mesh_length/2)}）"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

# ===========================
# 右パネル：拡大図
# ===========================
ax2 = axes[1]

z_lo, z_hi   = 1550, 2450  # 表示範囲（探触子全幅を包含）
x_depth_show = 35           # 材料内部の表示深さ [mesh]
ann_space    = 30           # 表面外側の注釈スペース [mesh]

ax2.set_xlim(z_lo, z_hi)
ax2.set_ylim(nx + ann_space, nx - x_depth_show)  # 大きい値が下（inverted y）
ax2.set_xlabel("z 方向 [mesh]  (1 mesh = 10 µm)", labelpad=6)
ax2.set_ylabel("x 方向 [mesh]", labelpad=6)
ax2.set_title("拡大図：くさび形プロファイルと探触子の位置\n"
              "（表面 = 実線、上方向 = 材料内部）")

# ---- 深さプロファイルの計算 ----
z_arr = np.arange(z_lo, z_hi + 1)
depth_arr = np.array([kusabi_depth(z) for z in z_arr], dtype=float)
surface_arr = nx - depth_arr   # 各 z での固体表面の x 座標

# ---- 金属ブロック（黄色 + 黒枠） ----
ax2.add_patch(mpatches.Rectangle(
    (z_lo, nx - x_depth_show), z_hi - z_lo, x_depth_show,
    facecolor="lightyellow", edgecolor="black", linewidth=2.0, zorder=1
))

# ---- くさび形プロファイル（白 = 空洞） ----
ax2.fill_between(z_arr, surface_arr, nx,
                 color="white", step="post", zorder=2)
# プロファイル輪郭（黒線）
ax2.step(z_arr, surface_arr, color="black", lw=1.2, zorder=4, where="post")

# ---- 表面ライン（x = nx） ----
ax2.axhline(y=nx, color="black", lw=2.0, zorder=5)

# ---- 探触子バー（表面外側の注釈スペース） ----
probe_bar_y = nx + 4
probe_bar_h = 6
ax2.add_patch(mpatches.Rectangle(
    (sz_l, probe_bar_y), sz_r - sz_l, probe_bar_h,
    facecolor="royalblue", edgecolor="black", lw=0.8, alpha=0.85, zorder=5,
    label=f"探触子  [{int(sz_l)}, {int(sz_r)}]"
))

# 探触子中心線
ax2.axvline(x=sz, color="royalblue", lw=1.5, ls="--", zorder=6,
            label=f"探触子中心 sz = {int(sz)}")

# ---- ピッチ境界（垂直の点線） ----
# 表示範囲内の全ピッチ境界
for i in range(z_lo // mn_p, z_hi // mn_p + 2):
    z_ps = i * mn_p
    if z_lo <= z_ps <= z_hi:
        ax2.axvline(x=z_ps, color="gray", lw=0.7, ls=":", alpha=0.6, zorder=3)

# ---- 注釈（表面外側） ----
y_ann1 = nx + 13  # 探触子径の矢印
y_ann2 = nx + 21  # ピッチの矢印

# (1) 探触子径
ax2.annotate("", xy=(sz_r, y_ann1), xytext=(sz_l, y_ann1),
             arrowprops=dict(arrowstyle="<->", color="royalblue", lw=2.0), zorder=7)
ax2.text(sz, y_ann1 + 1.5,
         f"探触子径: {probe_d*1e3:.1f} mm  ({int(sz_r - sz_l)} mesh)",
         ha="center", va="top", fontsize=9, color="royalblue")

# (2) ピッチ（探触子中心が含まれるピッチ）
i_center = sz // mn_p
z_ps = i_center * mn_p
z_pe = (i_center + 1) * mn_p
ax2.annotate("", xy=(z_pe, y_ann2), xytext=(z_ps, y_ann2),
             arrowprops=dict(arrowstyle="<->", color="purple", lw=2.0), zorder=7)
ax2.text((z_ps + z_pe) / 2, y_ann2 + 1.5,
         f"pitch: {f_pitch*1e3:.2f} mm  ({mn_p} mesh)",
         ha="center", va="top", fontsize=9, color="purple")

# ---- 注釈（材料内部） ----
# (3) 最大深さ矢印（ピッチ先頭の垂直壁）
# 直前ピッチの末端（depth≈0の位置）から1つ手前に配置
z_wall = z_ps   # 垂直壁の位置
z_arr_pos = z_wall - 15  # 矢印を壁の左側（前ピッチ末端付近）に配置
ax2.annotate("", xy=(z_arr_pos, nx), xytext=(z_arr_pos, nx - mn_d),
             arrowprops=dict(arrowstyle="<->", color="sienna", lw=1.8), zorder=6)
ax2.text(z_arr_pos - 8, nx - mn_d / 2,
         f"depth:\n{f_depth*1e3:.2f} mm",
         ha="right", va="center", fontsize=8.5, color="sienna")

# (4) 斜面の傾きラベル（代表1ピッチ内に注記）
z_mid_slope = z_ps + mn_p // 2
x_mid_slope = nx - kusabi_depth(z_mid_slope) - 8
if x_mid_slope > nx - x_depth_show:
    ax2.text(z_mid_slope, x_mid_slope,
             f"slope\n({mn_d} mesh\n/ {mn_p} mesh)",
             ha="center", va="bottom", fontsize=7.5, color="saddlebrown",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

# ---- 凡例 ----
ax2.legend(loc="upper left", fontsize=8.5)

# ---- y 軸ラベル ----
ax2.set_yticks([nx - x_depth_show, nx - mn_d, nx, nx + ann_space])
ax2.set_yticklabels([
    f"{nx - x_depth_show}\n(材料内)",
    f"{nx - mn_d}\n(最大深さ)",
    f"{nx}\n(表面)",
    f"{nx + ann_space}\n(空間側)"
], fontsize=7.5)

# ---- x 軸ラベル ----
xticks = np.arange(1600, 2500, 100)
ax2.set_xticks(xticks)
ax2.set_xticklabels([f"{v}\n({v * mesh_length * 1e3:.1f}mm)" for v in xticks], fontsize=7.5)

plt.tight_layout()
plt.savefig("probe_placement_kusabi.png", dpi=150, bbox_inches="tight")
print("saved: probe_placement_kusabi.png")
plt.show()
