import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================== 1. パラメータ設定 ==================
# kusabi_sim.py のパラメータを継承
f_pitch = 1.25e-3    # ピッチ p [m]
f_depth = 0.20e-3    # 深さ d [m]
mesh_length = 1.0e-5 # メッシュサイズ [m]
step_size = 1       # 階段の高さ（メッシュ数）

x_length = 0.02
z_length = 0.04

nx = int(round(x_length / mesh_length))
nz = int(round(z_length / mesh_length))

# ================== 2. 関数定義 ==================
def isfree_kusabi_viz(nx, nz, f_pitch, f_depth, mesh_length, step_size):
    
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))
    
    cut_top_map = np.full(nz, nx) # 境界線描画用

    mn_p = int(round(f_pitch / mesh_length))
    mn_d = int(round(f_depth / mesh_length))

    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    num_f = int(np.ceil(nz / mn_p))

    for i in range(num_f):
        z_start = i * mn_p
        z_end   = min((i + 1) * mn_p, nz)

        for z in range(z_start, z_end):
            local_z = z - z_start
            ideal_depth = mn_d * (1.0 - (local_z) / mn_p)
            current_depth = (int(ideal_depth) // step_size) * step_size

            if current_depth > 0:
                cut_top = nx - current_depth
                cut_top = max(0, cut_top)

                T13_isfree[cut_top : nx + 1 , z] = 0
                T5_isfree[cut_top : nx, z] = 0
                T5_isfree[cut_top : nx, z+1] = 0
                
                cut_top_map[z] = cut_top
            else:
                cut_top_map[z] = nx

    return T13_isfree, T5_isfree, cut_top_map

def around_free(T13_isfree, T5_isfree):
    # =================================
    # Ux の処理は元のまま変更なし
    # =================================
    Ux_free_count = np.zeros((nx, nz), dtype=float)
    Uz_free_count = np.zeros((nx + 1, nz + 1), dtype=float)

    for i in range(nx):
        for j in range(nz):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    # =================================
    # Uz の処理を以下のように修正
    # =================================
    for i in range(nx + 1):
        for j in range(nz + 1):
            # 1. 境界の基本カウント (元の外枠条件を維持)
            if j == 0 or j == nz or i == 0 or i == nx:
                Uz_free_count[i, j] += 1
            
            # 2. elif を外し、配列範囲内にある周囲のノードの空洞をカウントする
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
T13_mask, T5_mask, cut_top_map = isfree_kusabi_viz(nx, nz, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count, Uz_free_count = around_free(T13_mask, T5_mask)

# ================== 4. プロット設定 (境界拡大) ==================
mn_p = int(round(f_pitch / mesh_length))
mn_d = int(round(f_depth / mesh_length))

# 見やすいように1ピッチ分＋少しの余白を切り出す
z_start = 0
z_end   = mn_p + 10
x_start = nx - mn_d - 10
x_end   = nx + 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')

# グリッド描画
ax.set_xticks(np.arange(z_start, z_end + 1, 5)) # 5目盛りごとにラベル
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
ax.set_title(f"Staggered Grid Layout (Step Size: {step_size} meshes)")

# 凡例
legend_elements = [
    Line2D([0], [0], color='k', lw=2.5, label='Boundary Shape'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',   markersize=8, label='T1/T3 Node (Solid)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='T5 Node (Solid)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=8, label='Ux Node (Active)'),
    Line2D([0], [0], marker='>', color='w', markerfacecolor='green',  markersize=8, label='Uz Node (Active)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()