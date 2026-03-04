##########################################################################
#  hanen_animation_with_zoom.py
#    - 半円（U字）状のきず FDTD
#    - 形状確認用の「拡大ズーム画像」を自動保存する機能を追加
#    - 【修正】ピッチ定義を図面仕様 (p = W + gap + W) に合わせ、実質1.0mmピッチに変更
##########################################################################

import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import time

# ================== 調整パラメータ ==================

# 出力先
# output_dir = r"C:/Users/cs16/Documents/Test_folder/tmp_output" 
output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"

# きずパラメータ（表の #9 などを想定）
f_pitch = 1.25e-3   # ピッチ p [m] (図面値)
f_depth = 0.20e-3   # 深さ d [m]
f_width = 0.25e-3   # 幅 w [m] (固定)

# ===================================================

# ---------------- 基本パラメータ ----------------
x_length = 0.02
y_length = 0.04
mesh_length = 1.0e-5
nx = int(x_length / mesh_length)
ny = int(y_length / mesh_length)

dx = x_length / nx
dy = y_length / ny

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))
ct = np.sqrt(G / rho)
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)

# ★キャリブレーション済みの周波数
f = 4.7e6   
wn = 3.5    

T = 1 / f
lam = cl / f
k = 1 / lam
n = T / dt

# ---------------- 半円（U字）マスク生成（ピッチ定義修正版） ----------------
def isfree_hanen(nx, ny, f_pitch, f_depth, f_width, mesh_length):
    # 1:固体 / 0:空洞
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree  = np.ones((nx, ny + 1))

    # 幅の離散化
    mn_w = max(1, int(round(f_width / mesh_length)))
    
    # 図面ピッチの離散化 (1.25mm)
    mn_p_val = max(1, int(round(f_pitch / mesh_length)))

    # ★修正箇所：図面定義 (p = W + gap + W) から gap を逆算
    # gap = p - 2W
    mn_nf = max(0, mn_p_val - 2 * mn_w)

    # 真の繰り返し間隔 (Period) = W + gap
    mn_period = mn_w + mn_nf

    # 半径と直管部の計算
    radius_m = f_width / 2.0
    mn_r = int(round(radius_m / mesh_length))

    straight_h_m = f_depth - radius_m
    mn_straight = int(round(straight_h_m / mesh_length))
    if mn_straight < 0: mn_straight = 0

    # 外枠設定
    T13_isfree[0, 0:ny]  = 0; T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0]   = 0; T5_isfree[0:nx, ny]  = 0

    # きずの本数（真の周期 mn_period で割る）
    num_f = int(np.ceil(ny / mn_period))

    for i in range(num_f):
        # きずの開始位置 (y)
        # 図面左端基準の場合、y=0 から「きず」が始まると仮定
        y_start = i * mn_period
        
        # きずの中心座標 (y) = 開始位置 + 半径相当
        center_y = y_start + mn_w // 2

        if center_y >= ny: break

        # 中心 center_y を基準に円（＋直管）を描画
        for x_local in range(-mn_r, mn_r + 1):
            y_pos = center_y + x_local
            if y_pos < 0 or y_pos >= ny: continue

            dist_sq = x_local**2
            r_sq    = mn_r**2
            
            if dist_sq < r_sq:
                # 円部分の深さ
                h_circle = int(np.sqrt(r_sq - dist_sq))
                # 全体の深さ
                total_h = mn_straight + h_circle
                
                if total_h > 0:
                    # 表面(nx)から total_h だけ掘る
                    cut_top = nx - total_h
                    cut_top = max(0, cut_top)
                    
                    # マスク適用
                    # T13: cut_top 〜 nx
                    # T5 : cut_top 〜 nx-1
                    T13_isfree[cut_top : nx + 1, y_pos] = 0
                    T5_isfree[cut_top : nx, y_pos] = 0

    return T13_isfree, T5_isfree

# ---------------- 【追加機能】形状拡大チェック画像の保存 ----------------
def save_geometry_check(T13_isfree, nx, ny, f_depth, f_width, mesh_length, output_dir):
    """
    作成されたマスクの一部を切り取って、きずの形状が正しいか確認する画像を保存する関数
    """
    # 拡大範囲の決定（中央付近のきずを探す）
    center_y = ny // 2
    mn_d = int(f_depth / mesh_length)
    mn_w = int(f_width / mesh_length)
    
    # 表示範囲: 幅はきずの3倍、深さはきずの1.5倍くらい
    margin_w = mn_w * 2
    margin_d = mn_d + 10
    
    y_start = max(0, center_y - margin_w)
    y_end   = min(ny, center_y + margin_w)
    x_start = max(0, nx - margin_d)
    x_end   = nx
    
    # マスクを切り出す (0:空洞, 1:固体)
    zoom_area = T13_isfree[x_start:x_end, y_start:y_end]
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 色設定: 白=空洞, グレー=固体
    cmap = ListedColormap(['white', '#CCCCCC'])
    
    # 描画 (origin='upper' で上を0行目にするが、FDTD的には下がnxなので下を底にする)
    # ここでは直感的に「画像の下が試験体の底」になるように表示します
    ax.imshow(zoom_area, cmap=cmap, origin='upper', aspect='equal')
    
    # グリッド線（ピクセル境界）
    ax.set_xticks(np.arange(-0.5, zoom_area.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, zoom_area.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    # タイトルと軸
    ax.set_title(f"Geometry Check (Zoomed)\nDepth={f_depth*1000}mm, Width={f_width*1000}mm\n(1 block = 0.01mm)", fontsize=12)
    ax.set_xlabel("Width direction (mesh)")
    ax.set_ylabel("Depth direction (mesh)")
    
    # 保存
    save_path = os.path.join(output_dir, "geometry_zoom_check_hanen.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"★形状チェック画像を保存しました → {save_path}")

# ---------------- 周囲カウント(Voxel法) ----------------
def around_free():
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

    for i in range(nx):
        for j in range(ny):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    for i in range(nx + 1):
        for j in range(ny + 1):
            if j == 0 or j == ny or i == 0 or i == nx:
                Uy_free_count[i, j] += 1
            elif 0 < i < nx and 0 < j < ny:
                if T13_isfree[i, j - 1] == 0: Uy_free_count[i, j] += 1
                if T13_isfree[i, j] == 0:     Uy_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0:  Uy_free_count[i, j] += 1
                if T5_isfree[i, j] == 0:      Uy_free_count[i, j] += 1

    return Ux_free_count, Uy_free_count

# ---------------- 入射波形 ----------------
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3

# ---------------- 計測設定 ----------------
sy = int(ny / 2)
sx = 0
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)

t_max = 4 * x_length / cl / dt 

# ---------------- 実行準備 ----------------
print(f"Shape: Semicircle/U-shape")
print(f"Pitch(Diagram) = {f_pitch*1000} mm -> Real Period = {(f_width + max(0, f_pitch - 2*f_width))*1000:.2f} mm")
print(f"Depth = {f_depth*1000} mm, Width = {f_width*1000} mm")

# 出力ディレクトリ作成
os.makedirs(output_dir, exist_ok=True)

T1 = cp.zeros((nx + 1, ny), dtype=float)
T3 = cp.zeros((nx + 1, ny), dtype=float)
T5 = cp.zeros((nx, ny + 1), dtype=float)
Ux = cp.zeros((nx, ny), dtype=float)
Uy = cp.zeros((nx + 1, ny + 1), dtype=float)
wave = np.zeros(int(t_max))

dtx = dt / dx
dty = dt / dy

# ★半円マスク生成
T13_isfree, T5_isfree = isfree_hanen(nx, ny, f_pitch, f_depth, f_width, mesh_length)

# ★【ここで形状チェック画像を保存します】
save_geometry_check(cp.asnumpy(T13_isfree), nx, ny, f_depth, f_width, mesh_length, output_dir)

# Voxelカウント
Ux_free_count, Uy_free_count = around_free()

T1_snaps = []
start_time = time.time()

# ---------------- 時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0: print(f"{t}/{int(t_max)}")

    # 境界条件
    T5[0:nx, 0] = 0; T5[0:nx, ny] = 0
    T3[0, 0:ny] = 0; T3[nx, 0:ny] = 0
    T1[nx, 0:ny] = 0; T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

    Uy[1:nx, 0]  -= (4/rho)*dtx * T3[1:nx, 0]
    Uy[1:nx, ny] -= (4/rho)*dtx * (-T3[1:nx, ny-1])
    Uy[nx, 1:ny] -= (4/rho)*dtx * (-T5[nx-1, 1:ny])

    # 応力更新
    T1[1:nx, :] -= dtx * (c11*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c13*(Uy[1:nx,1:] - Uy[1:nx,:-1]))
    T3[1:nx, :] -= dtx * (c13*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c11*(Uy[1:nx,1:] - Uy[1:nx,:-1]))
    T5[:, 1:ny] -= dtx * c55 * (Ux[:,1:] - Ux[:,:-1] + Uy[1:,1:ny] - Uy[:-1,1:ny])

    # きず内部の応力=0
    T1[T13_isfree == 0] = 0.0
    T3[T13_isfree == 0] = 0.0
    T5[T5_isfree[0:nx, :] == 0] = 0.0

    # 音源
    if t < int(len(wave4)):
        T1[0, sy_l:sy_r] = wave4[t]
    else:
        Uy[0, sy_l:sy_r] = 0; Ux[0, sy_l:sy_r] = 0
        T1[0, 0:ny] = 0; T5[0, 0:ny] = 0

    # 速度更新
    Ux[0:nx, 0:ny] = cp.where(
        cp.asarray(Ux_free_count) < 4,
        Ux - (4/rho/(4 - cp.asarray(Ux_free_count))) * dtx * (
            T1[1:nx+1, :] - T1[0:nx, :] + T5[:, 1:ny+1] - T5[:, 0:ny]
        ), 0
    )
    Uy[1:nx, 1:ny] = cp.where(
        cp.asarray(Uy_free_count[1:nx, 1:ny]) < 4,
        Uy[1:nx, 1:ny] - (4/rho/(4 - cp.asarray(Uy_free_count[1:nx, 1:ny]))) * dty * (
            T3[1:nx, 1:ny] - T3[1:nx, :-1] + T5[1:nx, 1:ny] - T5[:-1, 1:ny]
        ), 0
    )

    cp.cuda.Device().synchronize()

    if t > 0:
        wave[t] = cp.mean(T1[1, sy_l:sy_r])
        if t % 50 == 0:
            T1_snaps.append(cp.asnumpy(T1.copy()))

wave = cp.asnumpy(wave)
end_time = time.time()
print(f"Done. Time: {end_time - start_time:.2f} s")

# ---------------- 保存 & 表示 ----------------
csv_name = f"hanen_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.csv"
np.savetxt(os.path.join(output_dir, csv_name), wave, delimiter=',')
print(f"CSV保存完了: {csv_name}")

'''
# アニメーション
T1_all = np.array(T1_snaps)
t1_max = np.max(np.abs(T1_all))
fig, ax = plt.subplots()
im = ax.imshow(T1_snaps[0], cmap='jet', vmin=-t1_max, vmax=t1_max, animated=True)
ax.set_title("Semicircle/U-shape Groove")
plt.colorbar(im, ax=ax)

def update(f):
    im.set_array(T1_snaps[f])
    ax.set_title(f"Time: {f*50}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(T1_snaps), interval=100)
ani_path = os.path.join(output_dir, f"hanen_ani.mp4")
ani.save(ani_path, fps=15)
print(f"アニメーション保存完了: {ani_path}")
plt.show()

'''

# 波形
plt.figure(figsize=(10,4))
plt.plot(wave)
plt.title("Waveform at Probe (U-shape)")
plt.show()