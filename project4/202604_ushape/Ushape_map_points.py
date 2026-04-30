import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import japanize_matplotlib

output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

# ====== パラメータ（ushape_sim.py と統一） ======
mesh_length = 1.0e-5
nx = 2000
nz = 4000

f_width = 0.25e-3    # 溝幅 w [m]（固定）
f_depth = 0.20e-3    # 全深さ d [m]
f_pitch = 2.00e-3    # ピッチ P=W+Gap [m]
step_size = 1

mn_w = int(round(f_width / mesh_length))
if mn_w % 2 == 0:
    mn_w -= 1
mn_d  = int(round(f_depth / mesh_length))
mn_r  = mn_w // 2
mn_straight = mn_d - mn_r

mn_p      = int(round(f_pitch / mesh_length))
mn_nf     = max(0, mn_p - mn_w)        # Gap = P - W
mn_period = mn_w + mn_nf               # 繰り返し周期

# ====== nz/2 付近の溝を特定（ushape_sim.py と同式） ======
i_near          = max(0, int(nz / 2) // mn_period)
z_groove_start  = i_near * mn_period
z_groove_center = (z_groove_start + min(z_groove_start + mn_w, nz)) // 2
z_groove_end    = z_groove_start + mn_w   # 最初の溝外グリッド

# ====== 溝2（次の溝）の座標 ======
z2_start  = z_groove_start + mn_period
z2_center = (z2_start + min(z2_start + mn_w, nz)) // 2
z2_end    = z2_start + mn_w

# 探触子位置（隙間中心固定）
sz = (z_groove_end + z2_start) // 2

# ====== 深さ計算（ushape_sim.py の isfree_u_shape と同一ロジック） ======
def _depth_from_center(dz):
    """dz=|z-z_center| から最大深さを返す（内部用）"""
    max_d = 0
    for d in range(mn_d):
        d_step = (d // step_size) * step_size
        if d_step < mn_straight:
            width_at_d = mn_w
        else:
            arc_d = d_step - mn_straight
            if arc_d < mn_r:
                half_w_float = np.sqrt(mn_r**2 - arc_d**2)
                width_at_d = int(half_w_float * 2)
            else:
                width_at_d = 0
            if width_at_d > 0 and width_at_d % 2 == 0:
                width_at_d -= 1
            if width_at_d < 0:
                width_at_d = 0
        half = width_at_d // 2
        if dz <= half:
            max_d = d + 1
    return max_d

def ushape_depth_exact(z):
    """z グリッドでの最大深さ（溝1・溝2 両方チェック、溝-firstに対応）"""
    best = 0
    for ig in [i_near, i_near + 1]:
        zs = ig * mn_period
        ze = zs + mn_w
        zc = (zs + min(ze, nz)) // 2
        best = max(best, _depth_from_center(abs(int(z) - zc)))
    return best

# ====== 表示範囲：溝1・溝2 を両方収める ======
margin = 15
z_lo   = z_groove_start - margin
z_hi   = z2_end          + margin

z_arr     = np.arange(z_lo, z_hi + 1)
depth_arr = np.array([ushape_depth_exact(z) for z in z_arr], dtype=float)
surf_arr  = nx - depth_arr

# ====== 凸角（計測点）: 溝1右辺 + 溝2左辺 ======
def groove_corners(zs, ze, zc, keep_side='both'):
    """溝1つ分の凸角 (z, x) リストを返す。
    keep_side: 'both' / 'right'（隙間側=右辺のみ）/ 'left'（隙間側=左辺のみ）
    """
    czs, cxs = [], []
    zl = zs - 1
    zr = ze + 1
    depths = {z: _depth_from_center(abs(z - zc)) for z in range(zl, zr + 1)}

    for z in range(zl, zr):
        d_cur  = depths.get(z,     0)
        d_next = depths.get(z + 1, 0)
        if d_cur == d_next:
            continue
        z_pos = z + 1
        if keep_side == 'right' and z_pos <= zc: continue
        if keep_side == 'left'  and z_pos >  zc: continue
        if d_cur == 0:
            czs.append(z_pos); cxs.append(nx)
            czs.append(z_pos); cxs.append(nx - d_next)
        elif d_next == 0:
            czs.append(z_pos); cxs.append(nx)
            czs.append(z_pos); cxs.append(nx - d_cur)
        else:
            czs.append(z_pos); cxs.append(nx - max(d_cur, d_next))
    return czs, cxs

cz1, cx1 = groove_corners(z_groove_start, z_groove_end, z_groove_center, keep_side='right')
cz2, cx2 = groove_corners(z2_start,       z2_end,       z2_center,       keep_side='left')

# groove_corners は (x=2000, x=1992) の順に生成するが、Point 8=x=1992, Point 9=x=2000 に合わせて入れ替え
cx1[-2], cx1[-1] = cx1[-1], cx1[-2]
cz1[-2], cz1[-1] = cz1[-1], cz1[-2]

corner_z  = np.array(cz1 + cz2)
corner_x  = np.array(cx1 + cx2)

# ====== 描画設定 ======
x_depth_show = mn_d + 10
ann_space    = 18

fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle("U字型表面の計測点の位置確認\n"
             f"（pitch={f_pitch*1e3:.2f} mm,  width={f_width*1e3:.2f} mm,  "
             f"depth={f_depth*1e3:.2f} mm）",
             fontsize=11)

ax.set_xlim(z_lo, z_hi)
ax.set_ylim(nx + ann_space, nx - x_depth_show)  # 大きい値が下（表面）
ax.set_xlabel("z 方向 [mm]", fontsize=11, labelpad=6)
ax.set_ylabel("x 方向 [mm]", fontsize=11, labelpad=6)
ax.set_aspect('equal')

# ---- 材料ブロック（背景） ----
ax.add_patch(mpatches.Rectangle(
    (z_lo, nx - x_depth_show), z_hi - z_lo, x_depth_show,
    facecolor="lightyellow", edgecolor="black", lw=2.0, zorder=1
))

# ---- U字型プロファイル（階段形状・ushape_sim.py と同一） ----
ax.fill_between(z_arr, surf_arr, nx, color="white", step="post", zorder=2)
ax.step(z_arr, surf_arr, color="black", lw=1.5, where="post", zorder=4)

# ---- 表面ライン（平坦部） ----
ax.axhline(y=nx, color="black", lw=2.0, zorder=5)

# ---- 探触子中心ライン ----
ax.axvline(x=sz, color="royalblue", lw=1.2, ls="--", alpha=0.9, zorder=6,
           label=f"探触子中心  z={sz * mesh_length * 1e3:.2f} mm")

# ---- 計測点（溝入口コーナー） ----
ax.scatter(corner_z, corner_x, s=60, color="red", marker="o", zorder=7,
           label="計測点")

# ---- Point 番号アノテーション ----
for i, (cz, cx) in enumerate(zip(corner_z, corner_x)):
    ax.text(cz + 1, cx - 1, f"{i+1}", fontsize=8, color="red",
            ha="left", va="bottom", zorder=9)

# ---- アノテーション：溝幅 ----
y_w = nx - mn_straight - 1
ax.annotate("", xy=(z_groove_end, y_w), xytext=(z_groove_start, y_w),
            arrowprops=dict(arrowstyle="<->", color="green", lw=1.5), zorder=8)
ax.text(z_groove_center, y_w - 1,
        f"w = {f_width*1e3:.2f} mm",
        ha="center", va="bottom", fontsize=12, color="green")

# ---- アノテーション：全深さ ----
ax.annotate("", xy=(z_groove_center, nx - mn_d), xytext=(z_groove_center, nx),
            arrowprops=dict(arrowstyle="<->", color="sienna", lw=1.5), zorder=8)
ax.text(z_groove_start - 3, nx - mn_d / 2,
        f"d = {f_depth*1e3:.2f} mm",
        ha="right", va="center", fontsize=12, color="sienna")

# ---- アノテーション：直線部深さ（点線） ----
if mn_straight > 0:
    ax.axhline(y=nx - mn_straight, color="gray", lw=1.0, ls=":", zorder=3)
    ax.text(z_hi - 1, nx - mn_straight - 1,
            f"直線部  {mn_straight * mesh_length * 1e3:.3f} mm",
            ha="right", va="bottom", fontsize=9, color="gray")

# ---- アノテーション：半円半径 ----
ax.text(z_groove_center + 1, nx - mn_straight - mn_r // 2,
        f"R = {mn_r * mesh_length * 1e3:.3f} mm",
        ha="left", va="center", fontsize=9, color="navy")

# ---- アノテーション：ピッチ P=W+Gap（溝1左端〜溝2左端） ----
y_ann = nx + 12
ax.annotate("",
            xy=(z2_start, y_ann), xytext=(z_groove_start, y_ann),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5), zorder=8)
ax.text((z_groove_start + z2_start) / 2, y_ann + 1.5,
        f"pitch P = {f_pitch * 1e3:.2f} mm ",
        ha="center", va="top", fontsize=11, color="purple")

# ---- 凡例 ----
ax.legend(loc="upper right", fontsize=11)

# ---- 軸ラベル（mm 表記） ----
yticks = [nx - x_depth_show, nx - mn_d, nx - mn_straight, nx]
yticklabels = [
    f"{(nx - x_depth_show) * mesh_length * 1e3:.2f}",
    f"{(nx - mn_d) * mesh_length * 1e3:.2f}\n(最大深さ)",
    f"{(nx - mn_straight) * mesh_length * 1e3:.2f}\n(直線部底端)",
    f"{nx * mesh_length * 1e3:.2f}\n(表面)",
]
if mn_straight == 0:
    yticks = [nx - x_depth_show, nx - mn_d, nx]
    yticklabels = [
        f"{(nx - x_depth_show) * mesh_length * 1e3:.2f}",
        f"{(nx - mn_d) * mesh_length * 1e3:.2f}\n(最大深さ)",
        f"{nx * mesh_length * 1e3:.2f}\n(表面)",
    ]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=11)

xtick_step = max(5, (z_hi - z_lo) // 8)
xticks = np.arange(z_lo, z_hi + 1, xtick_step)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{v * mesh_length * 1e3:.2f}" for v in xticks], fontsize=11)

# ---- グリッド線（1 メッシュ単位） ----
for _z in range(z_lo, z_hi + 1):
    ax.axvline(x=_z, color="lightgray", linewidth=0.3, zorder=3)
for _x in range(nx - x_depth_show, nx + ann_space + 1):
    ax.axhline(y=_x, color="lightgray", linewidth=0.3, zorder=3)

plt.tight_layout(rect=[0, 0, 1, 0.92])

fig_name = os.path.join(
    output_dir,
    f"ushape_map_points_pitch{int(f_pitch*1e5)}_"
    f"width{int(f_width*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(fig_name, dpi=150, bbox_inches="tight")
print(f"saved: {fig_name}")
plt.show()
