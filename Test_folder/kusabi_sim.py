import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time

# ================== 調整パラメータ ==================

# 出力先
# output_dir = r"C:/Users/cs16/Roughness/Test_folder/tmp_output"  # 研究室PC
output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"   # 自宅PC

# ★くさび形(IMG_2910) のパラメータ
f_pitch = 1.25e-3   # ピッチ p [m]
f_depth = 0.20e-3   # 深さ d [m]

# ★追加パラメータ: 階段の高さ（メッシュ数）
step_size = 5

# ===================================================

# ---------------- 基本パラメータ ----------------
x_length = 0.02   # [m]
y_length = 0.04   # [m]
mesh_length = 1.0e-5  # メッシュサイズ [m]

nx = int(round(x_length / mesh_length))
ny = int(round(y_length / mesh_length))

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
f = 4.7e6  # 周波数
T = 1 / f
lam = cl / f
k = 1 / lam
n = T / dt

# ---------------- くさび形マスク生成（階段対応版） ----------------
def isfree_kusabi(nx, ny, f_pitch, f_depth, mesh_length, step_size):
    # 1:固体 / 0:空洞
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree  = np.ones((nx, ny + 1))

    # パラメータの離散化
    mn_p = int(round(f_pitch / mesh_length))  # 1ピッチのセル数
    mn_d = int(round(f_depth / mesh_length))  # 最大深さのセル数

    # 外枠の処理
    T13_isfree[0, 0:ny]  = 0
    T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, ny]  = 0

    # くさびの数
    num_f = int(np.ceil(ny / mn_p))

    for i in range(num_f):
        # 1つのくさびの開始位置と終了位置
        y_start = i * mn_p
        y_end   = min((i + 1) * mn_p, ny)

        # スロープを作るループ (y方向 = 幅方向)
        for y in range(y_start, y_end):
            # ピッチ内のローカル座標
            local_y = y - y_start
            
            # 理想的な深さを計算
            # 深い(mn_d) -> 浅い(0) へ直線的に変化
            ideal_depth = mn_d * (1.0 - (local_y) / mn_p)
            
            # ★ここで階段状にする
            # 理想の深さを step_size で割って切り捨て、また掛ける
            # 例: step=5, depth=19 -> 3 * 5 = 15
            current_depth = (int(ideal_depth) // step_size) * step_size
            
            # 底面(nx)から上に向かって削る
            if current_depth > 0:
                cut_top = nx - current_depth
                cut_top = max(0, cut_top)
                
                # 空洞に設定
                T13_isfree[cut_top : nx, y] = 0
                T5_isfree[cut_top : nx, y] = 0
                T5_isfree[cut_top : nx, y+1] = 0


    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 (Voxel法) ----------------
def around_free(T13_isfree, T5_isfree):
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

    # Ux 周囲カウント
    for i in range(nx):
        for j in range(ny):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    # Uy 周囲カウント
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
wn = 2.5
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
print(f"Pitch(p) = {f_pitch*1000} mm")
print(f"Depth(d) = {f_depth*1000} mm")
print(f"Step Size = {step_size} mesh(es)")

# 配列確保
T1 = cp.zeros((nx + 1, ny), dtype=float)
T3 = cp.zeros((nx + 1, ny), dtype=float)
T5 = cp.zeros((nx, ny + 1), dtype=float)
Ux = cp.zeros((nx, ny), dtype=float)
Uy = cp.zeros((nx + 1, ny + 1), dtype=float)
wave = np.zeros(int(t_max))

dtx = dt / dx
dty = dt / dy

# ★くさび形関数を呼び出し（引数に step_size 追加）
T13_isfree_np, T5_isfree_np = isfree_kusabi(nx, ny, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count_np, Uy_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

# ★GPUへ転送
T13_isfree = cp.asarray(T13_isfree_np)
T5_isfree = cp.asarray(T5_isfree_np)
Ux_free_count = cp.asarray(Ux_free_count_np)
Uy_free_count = cp.asarray(Uy_free_count_np)

start_time = time.time()

# ---------------- 時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0: 
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    # 境界条件 (反射壁)
    T5[0:nx, 0] = 0; T5[0:nx, ny] = 0
    T3[0, 0:ny] = 0; T3[nx, 0:ny] = 0
    T1[nx, 0:ny] = 0; T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

    # Uy境界
    Uy[1:nx, 0]  -= (4/rho)*dtx * T3[1:nx, 0]
    Uy[1:nx, ny] -= (4/rho)*dtx * (-T3[1:nx, ny-1])
    Uy[nx, 1:ny] -= (4/rho)*dtx * (-T5[nx-1, 1:ny])

    # 応力更新
    T1[1:nx, :] -= dtx * (c11*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c13*(Uy[1:nx,1:] - Uy[1:nx,:-1]))
    T3[1:nx, :] -= dtx * (c13*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c11*(Uy[1:nx,1:] - Uy[1:nx,:-1]))
    T5[:, 1:ny] -= dtx * c55 * (Ux[:,1:] - Ux[:,:-1] + Uy[1:,1:ny] - Uy[:-1,1:ny])

    # ★くさび内部の応力=0
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
        Ux_free_count < 4,
        Ux - (4/rho/(4 - Ux_free_count)) * dtx * (
            T1[1:nx+1, :] - T1[0:nx, :] + T5[:, 1:ny+1] - T5[:, 0:ny]
        ), 0
    )
    Uy[1:nx, 1:ny] = cp.where(
        Uy_free_count[1:nx, 1:ny] < 4,
        Uy[1:nx, 1:ny] - (4/rho/(4 - Uy_free_count[1:nx, 1:ny])) * dty * (
            T3[1:nx, 1:ny] - T3[1:nx, :-1] + T5[1:nx, 1:ny] - T5[:-1, 1:ny]
        ), 0
    )

    if t > 0:
        wave[t] = cp.mean(T1[1, sy_l:sy_r])

    # 同期
    if t % 1000 == 0:
         cp.cuda.Device().synchronize()

wave = cp.asnumpy(wave)
end_time = time.time()
print(f"Done. Time: {end_time - start_time:.2f} s")

# ---------------- 保存 & 簡易表示 ----------------
os.makedirs(output_dir, exist_ok=True)

csv_name = f"kusabi2_cupy_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.csv"

# ファイル名に step_size を含める
# csv_name = f"kusabi_cupy_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}_step{step_size}.csv"

save_path = os.path.join(output_dir, csv_name)
np.savetxt(save_path, wave, delimiter=',')
print(f"Saved to: {save_path}")

plt.figure(figsize=(10,4))
plt.plot(wave)
plt.title(f"Waveform (Step Size={step_size})")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.show()