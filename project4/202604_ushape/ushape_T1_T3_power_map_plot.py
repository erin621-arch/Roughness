import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import japanize_matplotlib

# ================== 1. パラメータ設定 ==================
# ushape_sim.py のパラメータを継承
f_width = 0.25e-3    # 幅 w [m]
f_depth = 0.20e-3    # 深さ d [m]
f_pitch = 2.00e-3    # ピッチ p [m]
mesh_length = 1.0e-5 # メッシュサイズ [m]
step_size = 1        # 階段の高さ（メッシュ数）

x_length = 0.02
z_length = 0.04

nx = int(round(x_length / mesh_length))
nz = int(round(z_length / mesh_length))

# ================== 2. 関数定義 ==================
def isfree_ushape_viz(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size):
    """ushape_sim.pyのロジックに、可視化用の境界線マッピング(cut_top_map)を追加"""

    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx , nz + 1))
    
    cut_top_map = np.full(nz, nx)  # 境界線描画用

    mn_w = int(round(f_width / mesh_length))
    if mn_w % 2 == 0:
        mn_w -= 1

    mn_d = int(round(f_depth / mesh_length))
    mn_r = mn_w // 2
    mn_straight = mn_d - mn_r

    mn_p_val = max(1, int(round(f_pitch / mesh_length)))
    mn_nf = max(0, mn_p_val - mn_w)
    mn_period = mn_w + mn_nf

    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    num_f = int(np.ceil(nz / mn_period)) + 1

    for i in range(num_f):
        z_s = i * mn_period
        if z_s >= nz: break

        z_e = min(z_s + mn_w, nz)
        z_center = (z_s + z_e) // 2

        for d in range(mn_d):
            xi = (nx - 1) - d
            if xi < 0: break

            d_step = (d // step_size) * step_size
            width_at_d = 0

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
            zl = max(z_center - half, 0)
            zr = min(z_center + half + 1, nz)

            if zl < zr:
                T5_isfree[xi, zl:zr+1] = 0
                if xi < nx + 1:
                    T13_isfree[xi, zl:zr] = 0
                if xi + 1 < nx + 1:
                    T13_isfree[xi + 1, zl:zr] = 0

    # T5境界補正: T5[ix, iz] は T13[ix, iz] が空洞のときのみ void とする
    # (kusabi版と同じ規則: ix+1 側の T13 は参照しない)
    void_adj = np.zeros((nx, nz + 1), dtype=bool)
    void_adj[:, 1:]  |= (T13_isfree[:nx, :] == 0)  # T13[ix, iz-1] が空洞
    void_adj[:, :nz] |= (T13_isfree[:nx, :] == 0)  # T13[ix, iz  ] が空洞
    T5_isfree[void_adj] = 0

    # cut_top_map: 各z列で最初に空洞になるx index (外枠x=0は除外)
    for iz in range(nz):
        for ix in range(1, nx + 1):
            if T13_isfree[ix, iz] == 0:
                cut_top_map[iz] = ix
                break

    return T13_isfree, T5_isfree, cut_top_map

def around_free(T13_isfree, T5_isfree):
    Ux_free_count = np.zeros((nx, nz), dtype=float)
    Uz_free_count = np.zeros((nx + 1, nz + 1), dtype=float)

    for i in range(nx):
        for j in range(nz):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    for i in range(nx + 1):
        for j in range(nz + 1):
            if j == 0 or j == nz or i == 0 or i == nx:
                Uz_free_count[i, j] += 1

            if j > 0 and i < nx + 1:
                if T13_isfree[i, j - 1] == 0: Uz_free_count[i, j] += 1
            if j < nz and i < nx + 1:
                if T13_isfree[i, j] == 0:     Uz_free_count[i, j] += 1
            if i > 0 and j < nz + 1:
                if T5_isfree[i - 1, j] == 0:  Uz_free_count[i, j] += 1
            if i < nx and j < nz + 1:
                if T5_isfree[i, j] == 0:      Uz_free_count[i, j] += 1

    # ★後処理：4方向すべてが外枠または空洞のUzノードを非活性に強制
    dir1 = np.ones((nx + 1, nz + 1), dtype=bool)
    dir1[:, 1:] = (T13_isfree == 0)
    dir2 = np.ones((nx + 1, nz + 1), dtype=bool)
    dir2[:, :nz] = (T13_isfree == 0)
    dir3 = np.ones((nx + 1, nz + 1), dtype=bool)
    dir3[1:, :] = (T5_isfree == 0)
    dir4 = np.ones((nx + 1, nz + 1), dtype=bool)
    dir4[:nx, :] = (T5_isfree == 0)
    Uz_free_count[dir1 & dir2 & dir3 & dir4] = 4

    return Ux_free_count, Uz_free_count

# ================== 3. データ生成 ==================
print("データ生成中...")
T13_mask, T5_mask, cut_top_map = isfree_ushape_viz(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count, Uz_free_count = around_free(T13_mask, T5_mask)

# ================== 4. プロット設定 (境界拡大) ==================
mn_w = int(round(f_width / mesh_length))
if mn_w % 2 == 0:
    mn_w -= 1
mn_d = int(round(f_depth / mesh_length))

# 見やすいように1ピリオド分＋少しの余白を切り出す (kusabiと統一)
mn_p_val = max(1, int(round(f_pitch / mesh_length)))
mn_nf = max(0, mn_p_val - mn_w)
mn_period = mn_w + mn_nf

# ushape_surface_sim_center と同一のロジックで溝位置を特定
_sz_ref         = int(nz / 2)
_i_near         = max(0, _sz_ref // mn_period)
z_groove_start  = _i_near * mn_period
z_groove_end    = z_groove_start + mn_w
z_groove_center = (z_groove_start + z_groove_end) // 2

z_start = z_groove_start - 5
z_end   = z_groove_end + 10
x_start = nx - mn_d - 10
x_end   = nx + 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')

# グリッド描画
ax.set_xticks(np.arange(z_start, z_end + 1, 5))
ax.set_yticks(np.arange(x_start, x_end + 1, 5))
ax.grid(which='major', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# --- 境界線（階段状）の描画 ---
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

ax.plot(line_z, line_x, 'k-', linewidth=2.5, label='Boundary', zorder=1)

# --- ノード一括収集 ---
t13_solid_x, t13_solid_z = [], []
t13_void_x,  t13_void_z  = [], []
t5_solid_x,  t5_solid_z  = [], []
t5_void_x,   t5_void_z   = [], []
ux_active_x, ux_active_z = [], []
uz_active_x, uz_active_z = [], []

for ix in range(x_start, x_end + 1):
    for iz in range(z_start, z_end):
        # T13
        if ix < nx + 1 and iz < nz:
            if T13_mask[ix, iz] == 1:
                t13_solid_x.append(ix); t13_solid_z.append(iz)
            else:
                t13_void_x.append(ix);  t13_void_z.append(iz)

        # T5
        if ix < nx and iz < nz + 1:
            if T5_mask[ix, iz] == 1:
                t5_solid_x.append(ix + 0.5); t5_solid_z.append(iz - 0.5)
            else:
                t5_void_x.append(ix + 0.5);  t5_void_z.append(iz - 0.5)

        # Ux
        if ix < nx and iz < nz:
            if Ux_free_count[ix, iz] < 4:
                ux_active_x.append(ix + 0.5); ux_active_z.append(iz)

        # Uz
        if ix < nx + 1 and iz < nz + 1:
            if Uz_free_count[ix, iz] < 4:
                uz_active_x.append(ix); uz_active_z.append(iz - 0.5)

# --- プロット実行 ---
m_size = 40

# T13 (丸)
ax.scatter(t13_solid_z, t13_solid_x, s=m_size, c='blue', marker='o', alpha=0.7, edgecolors='none')
ax.scatter(t13_void_z,  t13_void_x,  s=m_size, c='lightgray', marker='o', alpha=0.3, edgecolors='none')

# T5 (四角)
ax.scatter(t5_solid_z, t5_solid_x, s=m_size, c='purple', marker='s', alpha=0.7, edgecolors='none')
ax.scatter(t5_void_z,  t5_void_x,  s=m_size, c='lightgray', marker='s', alpha=0.3, edgecolors='none')

# Ux（x方向）: 下向き矢印 'v'
ax.scatter(ux_active_z, ux_active_x, s=m_size, c='orange', marker='v', alpha=0.9, edgecolors='none')

# Uz（z方向）: 右向き矢印 '>'
ax.scatter(uz_active_z, uz_active_x, s=m_size, c='green', marker='>', alpha=0.9, edgecolors='none')

# レイアウト（縦軸=x：下向きが+）
ax.set_xlim(z_start - 2, z_end + 2)
ax.set_ylim(x_end + 2, x_start - 2)

ax.set_xlabel("z index (Width Direction →)")
ax.set_ylabel("x index (Depth Direction ↓)")
ax.set_title(f"Staggered Grid Layout - Ushape (Step Size: {step_size} meshes)")

# ================== 5. 計測点・T1 測定位置 ==================
mn_r        = mn_w // 2
mn_straight = mn_d - mn_r

def _depth_fc(dz_abs):
    max_d = 0
    for d in range(mn_d):
        d_step = (d // step_size) * step_size
        if d_step < mn_straight:
            width_at_d = mn_w
        else:
            arc_d = d_step - mn_straight
            width_at_d = int(np.sqrt(mn_r**2 - arc_d**2) * 2) if arc_d < mn_r else 0
            if width_at_d > 0 and width_at_d % 2 == 0: width_at_d -= 1
            if width_at_d < 0: width_at_d = 0
        if dz_abs <= width_at_d // 2:
            max_d = d + 1
    return max_d

def _groove_corners_r(zs, ze, zc):
    czs, cxs = [], []
    depths = {z: _depth_fc(abs(z - zc)) for z in range(zs - 1, ze + 2)}
    for z in range(zs - 1, ze):
        d_cur, d_next = depths.get(z, 0), depths.get(z + 1, 0)
        if d_cur == d_next: continue
        z_pos = z + 1
        if z_pos <= zc: continue
        if d_cur == 0:
            czs += [z_pos, z_pos]; cxs += [nx, nx - d_next]
        elif d_next == 0:
            czs += [z_pos, z_pos]; cxs += [nx, nx - d_cur]
        else:
            czs.append(z_pos); cxs.append(nx - max(d_cur, d_next))
    return np.array(czs, dtype=int), np.array(cxs, dtype=int)

# 溝1 のコーナー点（ushape_surface_sim_center と同一の溝位置）
_zs, _ze = z_groove_start, z_groove_end
_zc = z_groove_center
_cz, _cx = _groove_corners_r(_zs, _ze, _zc)

# valid フィルタ (x < nx)
_valid = (_cx < nx) & (_cx >= 0) & (_cz >= 0) & (_cz < nz)
_cz, _cx = _cz[_valid], _cx[_valid]
# Point 9: x=nx-1=1999 を末尾に追加
_cz = np.append(_cz, _ze)
_cx = np.append(_cx, nx - 1)

# 概念位置（赤丸●）: Pt 1–8 は自然座標、Pt 9 は境界 x=nx
pt_z_concept = list(_cz[:-1]) + [_ze]
pt_x_concept = list(_cx[:-1]) + [nx]
ax.scatter(pt_z_concept, pt_x_concept, s=80, c='red', marker='o', zorder=12)

# T1 読取位置（赤×）: Pt 1–8 は概念位置と同一、Pt 9 のみ x=nx-1
pt_z_t1 = list(_cz)          # Pts 1–8 の z + Pt9 の z (=_ze)
pt_x_t1 = list(_cx)          # Pts 1–8 の x + Pt9 の x (=nx-1)
ax.scatter(pt_z_t1, pt_x_t1, s=80, c='red', marker='x', linewidths=1.5, zorder=13)

# ラベル
for i in range(len(_cz) - 1):
    ax.annotate(f"Pt {i+1}\n({_cx[i]},{_cz[i]})", xy=(_cz[i], _cx[i]),
                xytext=(_cz[i] + 1.0, _cx[i] - 1.5),
                fontsize=6.5, color='darkred', zorder=14)
# Pt 9: 概念位置と T1 読取位置を両方記載
ax.annotate(f"Pt 9\n概念:({nx},{_ze})\nT1:({nx-1},{_ze})", xy=(_ze, nx),
            xytext=(_ze + 1.0, nx - 1.5),
            fontsize=6.5, color='darkred', zorder=14)

# 凡例
legend_elements = [
    Line2D([0], [0], color='k', lw=2.5, label='Boundary Shape'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',   markersize=8, label='T1/T3 Node (Solid)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='T5 Node (Solid)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=8, label='Ux Node (Active)'),
    Line2D([0], [0], marker='>', color='w', markerfacecolor='green',  markersize=8, label='Uz Node (Active)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8,
           label='計測点（概念位置）'),
    Line2D([0], [0], marker='x', color='red', markersize=8, lw=0,
           label='T1 読取位置 (Pt 1–8 は概念位置と同一)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()
