import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time

# ================== 調整パラメータ ==================

# 出力先
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"  # 研究室PC
# output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"   # 自宅PC

# ★くさび形(IMG_2910) のパラメータ
f_pitch = 1.25e-3   # ピッチ p [m]
f_depth = 0.20e-3   # 深さ d [m]

# ★追加パラメータ: 階段の高さ（メッシュ数）
step_size = 1

# ===================================================

# ---------------- 基本パラメータ ----------------
x_length = 0.02   # [m]
z_length = 0.04   # [m]
mesh_length = 1.0e-5  # メッシュサイズ [m]

nx = int(round(x_length / mesh_length))
nz = int(round(z_length / mesh_length))

dx = x_length / nx
dz = z_length / nz

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
def isfree_kusabi(nx, nz, f_pitch, f_depth, mesh_length, step_size):
    # 1:固体 / 0:空洞
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

    # パラメータの離散化
    mn_p = int(round(f_pitch / mesh_length))  # 1ピッチのセル数
    mn_d = int(round(f_depth / mesh_length))  # 最大深さのセル数

    # 外枠の処理
    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    # くさびの数
    num_f = int(np.ceil(nz / mn_p))

    for i in range(num_f):
        # 1つのくさびの開始位置と終了位置
        z_start = i * mn_p
        z_end   = min((i + 1) * mn_p, nz)

        # スロープを作るループ (z方向 = 幅方向)
        for z in range(z_start, z_end):
            # ピッチ内のローカル座標
            local_z = z - z_start

            # 理想的な深さを計算
            # 深い(mn_d) -> 浅い(0) へ直線的に変化
            ideal_depth = mn_d * (1.0 - (local_z) / mn_p)

            # ★ここで階段状にする
            current_depth = (int(ideal_depth) // step_size) * step_size

            # 底面(nx)から上に向かって削る
            if current_depth > 0:
                cut_top = nx - current_depth
                cut_top = max(0, cut_top)

                # 空洞に設定
                T13_isfree[cut_top : nx, z] = 0
                T5_isfree[cut_top : nx, z] = 0
                T5_isfree[cut_top : nx, z+1] = 0

    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 (Voxel法) ----------------
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

# ---------------- 入射波形 ----------------
wn = 2.5
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3

# ---------------- 計測設定 ----------------
sz = int(nz / 2)
sx = 0
probe_d = 0.007
sz_l = sz - int(probe_d / mesh_length / 2)
sz_r = sz + int(probe_d / mesh_length / 2)

t_max = 4 * x_length / cl / dt

# ---------------- 実行準備 ----------------
print(f"Pitch(p) = {f_pitch*1000} mm")
print(f"Depth(d) = {f_depth*1000} mm")
print(f"Step Size = {step_size} mesh(es)")

# 配列確保
T1 = cp.zeros((nx + 1, nz), dtype=float)
T3 = cp.zeros((nx + 1, nz), dtype=float)
T5 = cp.zeros((nx, nz + 1), dtype=float)
Ux = cp.zeros((nx, nz), dtype=float)
Uz = cp.zeros((nx + 1, nz + 1), dtype=float)
wave = np.zeros(int(t_max))

dtx = dt / dx
dtz = dt / dz

# ★くさび形関数を呼び出し
T13_isfree_np, T5_isfree_np = isfree_kusabi(nx, nz, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count_np, Uz_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

# ★GPUへ転送
T13_isfree = cp.asarray(T13_isfree_np)
T5_isfree = cp.asarray(T5_isfree_np)
Ux_free_count = cp.asarray(Ux_free_count_np)
Uz_free_count = cp.asarray(Uz_free_count_np)

start_time = time.time()

# ---------------- 時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0:
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    # 境界条件 (反射壁)
    T5[0:nx, 0] = 0; T5[0:nx, nz] = 0
    T3[0, 0:nz] = 0; T3[nx, 0:nz] = 0
    T1[nx, 0:nz] = 0; T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

    # Uz境界
    Uz[1:nx, 0]  -= (4/rho)*dtx * T3[1:nx, 0]
    Uz[1:nx, nz] -= (4/rho)*dtx * (-T3[1:nx, nz-1])
    Uz[nx, 1:nz] -= (4/rho)*dtx * (-T5[nx-1, 1:nz])

    # 応力更新
    T1[1:nx, :] -= dtx * (c11*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c13*(Uz[1:nx,1:] - Uz[1:nx,:-1]))
    T3[1:nx, :] -= dtx * (c13*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c11*(Uz[1:nx,1:] - Uz[1:nx,:-1]))
    T5[:, 1:nz] -= dtx * c55 * (Ux[:,1:] - Ux[:,:-1] + Uz[1:,1:nz] - Uz[:-1,1:nz])

    # ★くさび内部の応力=0
    T1[T13_isfree == 0] = 0.0
    T3[T13_isfree == 0] = 0.0
    T5[T5_isfree[0:nx, :] == 0] = 0.0

    # 音源
    if t < int(len(wave4)):
        T1[0, sz_l:sz_r] = wave4[t]
    else:
        Uz[0, sz_l:sz_r] = 0; Ux[0, sz_l:sz_r] = 0
        T1[0, 0:nz] = 0; T5[0, 0:nz] = 0

    # 速度更新
    Ux[0:nx, 0:nz] = cp.where(
        Ux_free_count < 4,
        Ux - (4/rho/(4 - Ux_free_count)) * dtx * (
            T1[1:nx+1, :] - T1[0:nx, :] + T5[:, 1:nz+1] - T5[:, 0:nz]
        ), 0
    )
    Uz[1:nx, 1:nz] = cp.where(
        Uz_free_count[1:nx, 1:nz] < 4,
        Uz[1:nx, 1:nz] - (4/rho/(4 - Uz_free_count[1:nx, 1:nz])) * dtz * (
            T3[1:nx, 1:nz] - T3[1:nx, :-1] + T5[1:nx, 1:nz] - T5[:-1, 1:nz]
        ), 0
    )

    if t > 0:
        wave[t] = cp.mean(T1[1, sz_l:sz_r])

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
