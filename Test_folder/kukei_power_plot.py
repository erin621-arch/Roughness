import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import japanize_matplotlib

# ================== 1. パラメータ設定 (kukei_ori.txtに基づく) ==================
# 矩形きず特有のパラメータ
f_width = 0.25e-3    # きずの幅 [m] (kukei_ori.txt source: 11)
f_pitch = 2.00e-3    # きずの間隔 [m] (kukei_ori.txt source: 11)
f_depth = 0.20e-3    # きずの深さ [m] (kukei_ori.txt source: 11)

mesh_length = 1.0e-5 # メッシュサイズ [m]

# 解析領域サイズ（x-z）
x_length = 0.02
z_length = 0.04
nx = int(round(x_length / mesh_length))
nz = int(round(z_length / mesh_length))

# ================== 2. 関数定義 ==================

def isfree_rect(nx, nz, f_pitch, f_depth, f_width, mesh_length):
    """
    矩形（Rectangular）形状の free/solid マスクを生成
    kukei_ori.txt のロジック をベースに作成
    軸：縦=x（下向きが正）、横=z（右向きが正）
    """
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

    # 離散点数の計算
    mn_w = int(round(f_width / mesh_length))          # きずの幅
    mn_p = int(round(f_pitch / mesh_length))          # 1ピッチ
    mn_d = int(round(f_depth / mesh_length))          # きずの深さ
    mn_nf = int(round((f_pitch - f_width) / mesh_length)) # きずのない部分 (source: 3)

    # 外枠の初期化 (source: 3, 4)
    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    # 境界線描画用マップ（初期値は底面 nx）
    cut_top_map = np.full(nz, nx)

    num_f = int(np.ceil(nz / mn_p)) # きずの数

    for i in range(num_f):
        # きずの開始・終了位置 (z方向)
        z_start_flaw = mn_nf + i * mn_p
        z_end_flaw   = (i + 1) * mn_p
        
        # 配列外参照防止
        if z_start_flaw >= nz:
            break
        z_end_true = min(z_end_flaw, nz)

        # === きず部分を消す (source: 4, 5, 15, 19) ===
        
        # T5: 矩形領域を0にする
        # kukei_ori: T5_isfree[nx - mn_d:nx, mn_nf... : (i+1)*mn_p] = 0
        T5_isfree[nx - mn_d : nx, z_start_flaw : z_end_true] = 0

        # T13: 矩形領域を0にする (横方向のインデックスが -1 されている点に注意)
        # kukei_ori: T13_isfree[nx - mn_d:nx + 1, mn_nf... : (i+1)*mn_p - 1] = 0
        T13_end_true = min(z_end_flaw - 1, nz) # -1 を適用
        if T13_end_true > z_start_flaw:
            T13_isfree[nx - mn_d : nx + 1, z_start_flaw : T13_end_true] = 0

        # === 境界線描画用マップの更新 ===
        # きずがある部分は、表面の高さが nx - mn_d になる
        cut_top_map[z_start_flaw : z_end_true] = nx - mn_d

    return T13_isfree, T5_isfree, cut_top_map


def around_free_all(nx, nz, T13_isfree, T5_isfree):
    """
    近傍がfreeでない点（<4）を “Active” とみなすためのカウント
    (plot4_annotated.py から変更なし)
    """
    Ux_free_count = np.zeros((nx, nz))
    Uz_free_count = np.zeros((nx + 1, nz + 1))

    # Ux（x方向成分）用
    for ix in range(nx):
        for iz in range(nz):
            if T13_isfree[ix, iz] == 0:     Ux_free_count[ix, iz] += 1
            if T13_isfree[ix + 1, iz] == 0: Ux_free_count[ix, iz] += 1
            if T5_isfree[ix, iz] == 0:      Ux_free_count[ix, iz] += 1
            if T5_isfree[ix, iz + 1] == 0:  Ux_free_count[ix, iz] += 1

    # Uz（z方向成分）用
    for ix in range(nx + 1):
        for iz in range(nz + 1):
            if iz == 0 or iz == nz or ix == 0 or ix == nx:
                Uz_free_count[ix, iz] += 4
            elif 0 < ix < nx and 0 < iz < nz:
                if T13_isfree[ix, iz - 1] == 0: Uz_free_count[ix, iz] += 1
                if T13_isfree[ix, iz] == 0:     Uz_free_count[ix, iz] += 1
                if T5_isfree[ix - 1, iz] == 0:  Uz_free_count[ix, iz] += 1
                if T5_isfree[ix, iz] == 0:      Uz_free_count[ix, iz] += 1

    return Ux_free_count, Uz_free_count


# ================== 3. データ生成 ==================
print("Generating Data (Rectangular Flaw)...")
# 矩形用の関数を呼び出し
T13_mask, T5_mask, cut_top_map = isfree_rect(nx, nz, f_pitch, f_depth, f_width, mesh_length)
Ux_free_count, Uz_free_count = around_free_all(nx, nz, T13_mask, T5_mask)

# ================== 4. プロット設定 ==================
mn_p = int(round(f_pitch / mesh_length))
mn_d = int(round(f_depth / mesh_length))
mn_nf = int(round((f_pitch - f_width) / mesh_length))

# プロット範囲の設定
# 矩形きずはピッチの後半(mn_nf以降)にあるため、そこを中心に表示
z_start = mn_nf - 5        # きずの少し手前から
z_end   = mn_p + 5         # 1ピッチ終了の少し後まで
x_start = nx - mn_d - 5    # きずの深さより少し上から
x_end   = nx + 1           # 底まで

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')

# グリッド
ax.set_xticks(np.arange(z_start, z_end + 1, 1))
ax.set_yticks(np.arange(x_start, x_end + 1, 1))
ax.grid(which='major', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

# 境界線（(z, x) で描く）
line_z, line_x = [], []
for iz in range(z_start, z_end + 1): # +1 to cover the edge
    if 0 <= iz < len(cut_top_map):
        c_top = cut_top_map[iz]
        if len(line_z) == 0:
            line_z.extend([iz - 0.5, iz + 0.5])
            line_x.extend([c_top, c_top])
        else:
            if line_x[-1] != c_top:
                # 垂直な壁を描く
                line_z.extend([iz - 0.5, iz - 0.5, iz + 0.5])
                line_x.extend([line_x[-1], c_top, c_top])
            else:
                line_z.append(iz + 0.5)
                line_x.append(c_top)

ax.plot(line_z, line_x, 'k-', linewidth=3, label='Boundary', zorder=1)

# --- ノード一括収集（座標は x,z で管理） ---
t13_solid_x, t13_solid_z = [], []
t13_void_x,  t13_void_z  = [], []
t5_solid_x,  t5_solid_z  = [], []
t5_void_x,   t5_void_z   = [], []
ux_active_x, ux_active_z = [], [] 
uz_active_x, uz_active_z = [], [] 

for ix in range(x_start, x_end + 1):
    for iz in range(z_start, z_end + 1):
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

        # Ux（x方向成分）: (x=ix+0.5, z=iz)
        if ix < nx and iz < nz:
            if Ux_free_count[ix, iz] < 4:
                ux_active_x.append(ix + 0.5); ux_active_z.append(iz)

        # Uz（z方向成分）: (x=ix, z=iz-0.5)
        if ix < nx + 1 and iz < nz + 1:
            if Uz_free_count[ix, iz] < 4:
                uz_active_x.append(ix); uz_active_z.append(iz - 0.5)

# --- プロット実行 ---
m_size = 80

# T13 (丸)
ax.scatter(t13_solid_z, t13_solid_x, s=m_size, c='blue', marker='o', alpha=0.6, edgecolors='none')
ax.scatter(t13_void_z,  t13_void_x,  s=m_size, c='lightgray', marker='o', alpha=0.3, edgecolors='none')

# T5 (四角)
ax.scatter(t5_solid_z, t5_solid_x, s=m_size, c='purple', marker='s', alpha=0.6, edgecolors='none')
ax.scatter(t5_void_z,  t5_void_x,  s=m_size, c='lightgray', marker='s', alpha=0.2, edgecolors='none')

# Ux（x方向）: 下向き矢印 'v'
ax.scatter(ux_active_z, ux_active_x, s=m_size, c='orange', marker='v', alpha=0.8, edgecolors='none')

# Uz（z方向）: 右向き矢印 '>'
ax.scatter(uz_active_z, uz_active_x, s=m_size, c='green', marker='>', alpha=0.8, edgecolors='none')

# レイアウト（縦軸=x：下向きが+）
ax.set_xlim(z_start - 1, z_end + 1)
ax.set_ylim(x_end + 0.5, x_start - 0.5)

ax.set_xlabel("z index (→ positive)")
ax.set_ylabel("x index (↓ positive)")
ax.set_title("Rectangular Flaw View (kukei_ori params)")

# 凡例
legend_elements = [
    Line2D([0], [0], color='k', lw=3, label='Boundary'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',   markersize=10, label='T13 (Solid)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=10, label='T5 (Solid)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=10, label='Ux (Active)'),
    Line2D([0], [0], marker='>', color='w', markerfacecolor='green',  markersize=10, label='Uz (Active)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()