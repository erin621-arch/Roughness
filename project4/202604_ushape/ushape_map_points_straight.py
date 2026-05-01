"""
Ushape 直線部（右壁）の連続記録位置を可視化するスクリプト
- 格子マップ上に直線部の範囲と記録予定位置を強調表示
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import japanize_matplotlib

# ================== パラメータ（ushape_surface_sim_center.py と同一） ==================
f_width     = 0.25e-3
f_depth     = 0.20e-3
f_pitch     = 2.00e-3
mesh_length = 1.0e-5
step_size   = 1

x_length = 0.02
z_length = 0.04
nx = int(round(x_length / mesh_length))   # 2000
nz = int(round(z_length / mesh_length))   # 4000

mn_w = int(round(f_width / mesh_length))
if mn_w % 2 == 0:
    mn_w -= 1
mn_d        = int(round(f_depth / mesh_length))
mn_r        = mn_w // 2
mn_straight = mn_d - mn_r   # 直線部の深さ（グリッド数）

mn_p_val  = int(round(f_pitch / mesh_length))
mn_nf     = max(0, mn_p_val - mn_w)
mn_period = mn_w + mn_nf

print(f"mn_w={mn_w}, mn_d={mn_d}, mn_r={mn_r}, mn_straight={mn_straight}")

# 溝位置
_sz_ref         = int(nz / 2)
i_near          = max(0, _sz_ref // mn_period)
z_groove_start  = i_near * mn_period
z_groove_end    = z_groove_start + mn_w
z_groove_center = (z_groove_start + z_groove_end) // 2

# 直線部の x 範囲: d=0(xi=nx-1)〜d=mn_straight-1(xi=nx-mn_straight)
x_straight_top = nx - 1            # d=0 (溝入口)
x_straight_bot = nx - mn_straight  # d=mn_straight-1 (直線部の最深行)
# アーク部: d=mn_straight〜mn_d-1, xi=nx-mn_straight-1〜nx-mn_d
x_arc_top = nx - mn_straight - 1   # アーク上端
x_arc_bot = nx - mn_d              # アーク下端

print(f"溝1: z=[{z_groove_start}, {z_groove_end})  center={z_groove_center}")
print(f"直線部: x=[{x_straight_bot}, {x_straight_top}]  (深さ {mn_straight} 行)")
print(f"アーク部: x=[{x_arc_bot}, {x_arc_top}]  (深さ {mn_d - mn_straight} 行)")
print(f"直線部右壁記録: x=[{x_straight_bot}, {x_straight_top}], z={z_groove_end}")

# ================== isfree 生成（可視化用） ==================
def isfree_ushape(nx, nz, mn_w, mn_d, mn_r, mn_straight, mn_period, step_size):
    T13_isfree = np.ones((nx + 1, nz), dtype=np.int8)
    T5_isfree  = np.ones((nx,     nz + 1), dtype=np.int8)

    T13_isfree[0,  :] = 0
    T13_isfree[nx, :] = 0
    T5_isfree[:, 0]   = 0
    T5_isfree[:, nz]  = 0

    num_f = int(np.ceil(nz / mn_period)) + 1
    for i in range(num_f):
        z_s = i * mn_period
        if z_s >= nz:
            break
        z_e      = min(z_s + mn_w, nz)
        z_center = (z_s + z_e) // 2

        for d in range(mn_d):
            xi     = (nx - 1) - d
            if xi < 0:
                break
            d_step = (d // step_size) * step_size
            if d_step < mn_straight:
                w = mn_w
            else:
                arc_d = d_step - mn_straight
                w = int(np.sqrt(mn_r**2 - arc_d**2) * 2) if arc_d < mn_r else 0
                if w > 0 and w % 2 == 0:
                    w -= 1
                if w < 0:
                    w = 0

            half = w // 2
            zl   = max(z_center - half, 0)
            zr   = min(z_center + half + 1, nz)
            if zl < zr:
                T5_isfree[xi, zl:zr + 1] = 0
                if xi < nx + 1:
                    T13_isfree[xi, zl:zr] = 0
                if xi + 1 < nx + 1:
                    T13_isfree[xi + 1, zl:zr] = 0

    void_adj = np.zeros((nx, nz + 1), dtype=bool)
    void_adj[:, 1:]  |= (T13_isfree[:nx, :] == 0)
    void_adj[:, :nz] |= (T13_isfree[:nx, :] == 0)
    T5_isfree[void_adj] = 0

    return T13_isfree, T5_isfree

T13, T5 = isfree_ushape(nx, nz, mn_w, mn_d, mn_r, mn_straight, mn_period, step_size)

# ================== 表示範囲（溝周辺を拡大） ==================
margin_z = 8
margin_x = 5
z_start = z_groove_start - margin_z
z_end   = z_groove_end   + margin_z
x_start = x_arc_bot      - margin_x
x_end   = nx + 1

# ================== 境界線生成 ==================
cut_top_map = np.full(nz, nx, dtype=int)
for iz in range(nz):
    for ix in range(1, nx + 1):
        if T13[ix, iz] == 0:
            cut_top_map[iz] = ix
            break

line_z, line_x = [], []
for iz in range(z_start, z_end):
    if 0 <= iz < len(cut_top_map):
        c_top = cut_top_map[iz]
        if len(line_z) == 0:
            line_z.extend([iz - 0.5, iz + 0.5])
            line_x.extend([c_top, c_top])
        else:
            if line_x[-1] != c_top:
                line_z.extend([iz - 0.5, iz - 0.5, iz + 0.5])
                line_x.extend([line_x[-1], c_top, c_top])
            else:
                line_z.append(iz + 0.5)
                line_x.append(c_top)

# ================== プロット ==================
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_aspect('equal')
ax.set_title(
    f'Ushape 直線部の可視化\n'
    f'(pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.3f} mm, '
    f'mn_straight={mn_straight}, mn_r={mn_r})',
    fontsize=11
)

# グリッド
ax.set_xticks(np.arange(z_start, z_end + 1, 2))
ax.set_yticks(np.arange(x_start, x_end + 1, 2))
ax.grid(which='major', color='lightgray', linestyle=':', linewidth=0.5, alpha=0.6)

# --- T13 ノード（丸） ---
t13_solid_z, t13_solid_x = [], []
t13_void_z,  t13_void_x  = [], []
for ix in range(x_start, x_end + 1):
    for iz in range(z_start, z_end):
        if ix <= nx and iz < nz:
            if T13[ix, iz] == 1:
                t13_solid_z.append(iz); t13_solid_x.append(ix)
            else:
                t13_void_z.append(iz);  t13_void_x.append(ix)

ax.scatter(t13_solid_z, t13_solid_x, s=30, c='steelblue', marker='o', alpha=0.5, zorder=2)
ax.scatter(t13_void_z,  t13_void_x,  s=30, c='lightgray', marker='o', alpha=0.3, zorder=2)

# --- 境界線 ---
ax.plot(line_z, line_x, 'k-', linewidth=2.5, zorder=5)

# ================== 直線部のハイライト ==================
# 直線部の void 内部を薄い緑でシェード
rect_straight = mpatches.FancyArrowPatch(
    (z_groove_start - 0.5, x_straight_bot - 0.5),
    (z_groove_end - 0.5,   x_straight_top + 0.5),
    arrowstyle='-',
)
# 直線部領域を塗りつぶし（溝の void 内部）
straight_patch = mpatches.Rectangle(
    (z_groove_start - 0.5, x_straight_bot - 0.5),
    mn_w,
    mn_straight,
    linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.25, zorder=3,
    label=f'直線部（d=0〜{mn_straight-1}）'
)
ax.add_patch(straight_patch)

# アーク部領域（溝の void 内部）を薄いオレンジでシェード
arc_patch = mpatches.Rectangle(
    (z_groove_center - mn_r - 0.5, x_arc_bot - 0.5),
    mn_w,
    mn_d - mn_straight,
    linewidth=2, edgecolor='orange', facecolor='wheat', alpha=0.25, zorder=3,
    label=f'アーク部（d={mn_straight}〜{mn_d-1}）'
)
ax.add_patch(arc_patch)

# ================== 直線部右壁の記録予定位置 ==================
straight_xs = np.arange(x_straight_bot, x_straight_top + 1)  # [1992..1999]
straight_zs = np.full_like(straight_xs, z_groove_end)         # z=2025

ax.scatter(straight_zs, straight_xs, s=120, c='red', marker='*',
           zorder=10, label=f'直線部右壁 T1 記録点\n(z={z_groove_end}, x={x_straight_bot}〜{x_straight_top})')

# ラベル（各記録点）
for i, (xi, zi) in enumerate(zip(straight_xs, straight_zs)):
    d_val = nx - 1 - xi
    ax.annotate(f'd={d_val}\n({xi},{zi})', xy=(zi, xi),
                xytext=(zi + 1.2, xi),
                fontsize=6.5, color='darkred', va='center', zorder=11)

# 直線部〜アーク境界線（x=x_arc_top=1991 の位置）
ax.axhline(y=x_arc_top + 0.5, color='darkorange', linestyle='--', linewidth=1.5,
           label=f'直線部/アーク境界 (x={x_arc_top + 1})')

# 下端境界（T1[2000,:]=0）
ax.axhline(y=nx, color='purple', linestyle=':', linewidth=1.5,
           label=f'下端境界 T1[{nx},:]=0')

# ================== 溝範囲のラベル ==================
ax.annotate('', xy=(z_groove_start, x_start - 1), xytext=(z_groove_end, x_start - 1),
            arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
ax.text((z_groove_start + z_groove_end) / 2, x_start - 2.5,
        f'mn_w={mn_w}', ha='center', fontsize=8, color='green')

ax.annotate('', xy=(z_groove_end + 1, x_straight_bot), xytext=(z_groove_end + 1, x_straight_top),
            arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
ax.text(z_groove_end + 3.5, (x_straight_bot + x_straight_top) / 2,
        f'mn_straight\n={mn_straight}', ha='left', fontsize=8, color='green', va='center')

# ================== 座標軸・凡例 ==================
ax.set_xlim(z_start - 3, z_end + 12)
ax.set_ylim(x_end + 3, x_start - 5)
ax.set_xlabel('z index（幅方向 →）', fontsize=10)
ax.set_ylabel('x index（深さ方向 ↓）', fontsize=10)

ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()
out_path = r'C:/Users/cs16/Roughness/project4/tmp_output/ushape_straight_section_viz.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'保存: {out_path}')
plt.show()
