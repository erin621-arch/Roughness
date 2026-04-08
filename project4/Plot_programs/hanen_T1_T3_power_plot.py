import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================== 1. パラメータ設定 ==================
# hanen_sim.py のパラメータを継承
f_width = 0.25e-3    # 幅 w [m]
f_depth = 0.20e-3    # 全深さ d [m]
f_pitch = 1.25e-3    # ピッチ
mesh_length = 1.00e-5 # メッシュサイズ [m]
step_size = 1        # 階段の高さ

x_length = 0.02
z_length = 0.04

nx = int(round(x_length / mesh_length))
nz = int(round(z_length / mesh_length))

# ================== 2. 関数定義 ==================
# ---------------- U字型(弾丸型)きず生成関数 ----------------
def isfree_u_shape(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size):
    # 1:固体 / 0:空洞
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

    # --- 1. 寸法の離散化 ---
    mn_w = int(round(f_width / mesh_length))
    if mn_w % 2 == 0:
        mn_w -= 1

    mn_d = int(round(f_depth / mesh_length))
    mn_r = mn_w // 2
    mn_straight = mn_d - mn_r

    mn_p_val = max(1, int(round(f_pitch / mesh_length)))
    mn_nf = max(0, mn_p_val - 2 * mn_w)
    mn_period = mn_w + mn_nf

    # 外枠
    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    num_f = int(np.ceil(nz / mn_period))

    for i in range(num_f):
        z_start = i * mn_period
        if z_start >= nz: break

        z_end = min(z_start + mn_w, nz)
        z_center = (z_start + z_end) // 2

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

    # ===== T5境界補正 (kusabi版を参考に修正) =====
    # T5_isfree[ix, iz] の周囲4T13ノードのいずれかが空洞であれば、
    # そのT5ノードも空洞(0)に設定する。
    # T5[ix, iz] の4隣接T13: T13[ix, iz-1], T13[ix+1, iz-1],
    #                         T13[ix, iz  ], T13[ix+1, iz  ]
    # (T13_isfree: shape (nx+1, nz), T5_isfree: shape (nx, nz+1))
    void_adj = np.zeros((nx, nz + 1), dtype=bool)

    # iz-1 側チェック (iz = 1..nz)
    void_adj[:, 1:]  |= (T13_isfree[:nx,  :] == 0)  # T13[ix,   iz-1]
    void_adj[:, 1:]  |= (T13_isfree[1:nx+1, :] == 0)  # T13[ix+1, iz-1]

    # iz 側チェック (iz = 0..nz-1)
    void_adj[:, :nz] |= (T13_isfree[:nx,  :] == 0)  # T13[ix,   iz  ]
    void_adj[:, :nz] |= (T13_isfree[1:nx+1, :] == 0)  # T13[ix+1, iz  ]

    T5_isfree[void_adj] = 0

    return T13_isfree, T5_isfree

def around_free(T13_isfree, T5_isfree):
    """バグ修正版の安定した境界判定ロジック"""
    Ux_free_count = np.zeros((nx, nz), dtype=float)
    Uz_free_count = np.zeros((nx + 1, nz + 1), dtype=float)

    # Ux
    for i in range(nx):
        for j in range(nz):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    # Uz (安全な独立チェック方式)
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
T13_mask, T5_mask = isfree_u_shape(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count, Uz_free_count = around_free(T13_mask, T5_mask)

# ================== 4. プロット設定 (境界拡大) ==================
mn_w = int(round(f_width / mesh_length))
mn_d = int(round(f_depth / mesh_length))

# 見やすいように、最初の1つのきず（U字）にズームイン
z_start = 0
z_end   = mn_w + 10
x_start = nx - mn_d - 5
x_end   = nx + 2

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')

# グリッド描画
ax.set_xticks(np.arange(z_start, z_end + 1, 2))
ax.set_yticks(np.arange(x_start, x_end + 1, 2))
ax.grid(which='major', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# --- ブロック状の境界線描画 (U字対応版) ---
# T13の空洞(0)と固体(1)の境界を探して線を引く
for ix in range(x_start, x_end):
    for iz in range(z_start, z_end):
        if ix < nx + 1 and iz < nz:
            if T13_mask[ix, iz] == 0:
                # 上が固体
                if ix > 0 and T13_mask[ix - 1, iz] == 1:
                    ax.hlines(ix - 0.5, iz - 0.5, iz + 0.5, colors='k', lw=3, zorder=1)
                # 下が固体
                if ix < nx and T13_mask[ix + 1, iz] == 1:
                    ax.hlines(ix + 0.5, iz - 0.5, iz + 0.5, colors='k', lw=3, zorder=1)
                # 左が固体
                if iz > 0 and T13_mask[ix, iz - 1] == 1:
                    ax.vlines(iz - 0.5, ix - 0.5, ix + 0.5, colors='k', lw=3, zorder=1)
                # 右が固体
                if iz < nz - 1 and T13_mask[ix, iz + 1] == 1:
                    ax.vlines(iz + 0.5, ix - 0.5, ix + 0.5, colors='k', lw=3, zorder=1)

# --- ノード一括収集 ---
t13_solid_x, t13_solid_z = [], []
t13_void_x,  t13_void_z  = [], []
t5_solid_x,  t5_solid_z  = [], []
t5_void_x,   t5_void_z   = [], []
ux_active_x, ux_active_z = [], []
uz_active_x, uz_active_z = [], []

for ix in range(x_start, x_end):
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
ax.set_xlim(z_start - 1, z_end + 1)
ax.set_ylim(x_end + 1, x_start - 1)

ax.set_xlabel("z index (Width Direction →)")
ax.set_ylabel("x index (Depth Direction ↓)")
ax.set_title(f"Staggered Grid Layout (U-Shape, Step Size: {step_size})")

# 凡例
legend_elements = [
    Line2D([0], [0], color='k', lw=3, label='Boundary Shape'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',   markersize=8, label='T1/T3 Node (Solid)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='T5 Node (Solid)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=8, label='Ux Node (Active)'),
    Line2D([0], [0], marker='>', color='w', markerfacecolor='green',  markersize=8, label='Uz Node (Active)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()
