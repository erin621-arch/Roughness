import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time

# ================== 調整パラメータ ==================

# 出力先
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"  # 研究室PC
# output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"   # 自宅PC

# きずパラメータ
f_width = 0.25e-3   # 幅 [m]
f_pitch = 1.25e-3   # ピッチ [m]
f_depth = 0.00e-3   # 深さ [m]

# ★追加: 階段の高さ（メッシュ数）
# 1 = 通常（最も滑らか）
# 2 = 2メッシュ分を1段とする
# 5 = 5メッシュ分を1段とする（カクカクになる）
step_size = 1

# =======================================================================

# ---------------- 基本パラメータ ----------------
x_length = 0.02   # x方向の長さ [m]
z_length = 0.04   # z方向の長さ [m]
mesh_length = 1.00e-5  # メッシュ長 [m]

nx = int(x_length / mesh_length)
nz = int(z_length / mesh_length)

dx = x_length / nx
dz = z_length / nz

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P波
ct = np.sqrt(G / rho)                                    # S波
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)  # 時間刻み
f = 4.7e6                  # 周波数
T = 1 / f
lam = cl / f
k = 1 / lam
n = T / dt

# ---------------- 三角きず生成関数（高さ可変・対称化版） ----------------
def isfree_symmetric(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size):
    # 1:通常 / 0:自由境界（ゼロにする領域）
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

    # --- 1. 幅の決定（対称性のため奇数に固定） ---
    mn_w = int(round(f_width / mesh_length))
    if mn_w % 2 == 0:
        mn_w -= 1  # 偶数なら1引いて奇数にする

    # --- 2. ピッチと隙間の計算 ---
    mn_p_val = max(1, int(round(f_pitch / mesh_length)))
    mn_nf = max(0, mn_p_val - 2 * mn_w)
    mn_period = mn_w + mn_nf

    # 外枠の境界条件設定
    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    # きずの数
    num_f = int(np.ceil(nz / mn_period))

    # --- 3. 深さのメッシュ数 ---
    mn_d = int(round(f_depth / mesh_length))

    for i in range(num_f):
        # 横方向の位置決め
        z_start = i * mn_period
        if z_start >= nz: break

        z_end = min(z_start + mn_w, nz)
        z_center = (z_start + z_end) // 2

        # 深さ方向のループ（底面から上へ）
        for d in range(mn_d):
            xi = (nx - 1) - d
            if xi < 0: break

            # --- ★改良ロジック: 任意の高さ(step_size)で階段を作る ---

            # 深さを step_size 単位で切り捨てる
            d_step = (d // step_size) * step_size

            # (A) 階段状になった深さ(d_step)を使って幅を計算
            raw_w = mn_w * (1.0 - (d_step) / mn_d)
            width_at_d = int(round(raw_w))

            # (B) 強制的に奇数にする（左右対称にするため）
            if width_at_d % 2 == 0:
                width_at_d -= 1

            # (C) 最低幅は1とする
            if width_at_d < 1:
                width_at_d = 1

            # --- 配列への適用 ---
            half = width_at_d // 2
            zl = max(z_center - half, 0)
            zr = min(z_center + half + 1, nz)

            if zl < zr:
                T5_isfree[xi, zl:zr] = 0
                if xi < nx + 1:
                    T13_isfree[xi, zl:zr] = 0

    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 ----------------
def around_free(T13_isfree, T5_isfree):
    Ux_free_count = np.zeros((nx, nz), dtype=float)
    Uz_free_count = np.zeros((nx + 1, nz + 1), dtype=float)

    # Ux セル
    for i in range(nx):
        for j in range(nz):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    # Uz ノード
    for i in range(nx + 1):
        for j in range(nz + 1):
            if j == 0 or j == nz: Uz_free_count[i, j] += 1
            if i == 0 or i == nx: Uz_free_count[i, j] += 1

            if 0 < i < nx and 0 < j < nz:
                if T13_isfree[i, j - 1] == 0: Uz_free_count[i, j] += 1
                if T13_isfree[i, j] == 0: Uz_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0: Uz_free_count[i, j] += 1
                if T5_isfree[i, j] == 0: Uz_free_count[i, j] += 1

    return Ux_free_count, Uz_free_count

# ---------------- 入射波形 ----------------
wn = 2.5
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3

# ---------------- 音源・計測設定 ----------------
sz = int(nz / 2)
sx = 0
probe_d = 0.007
sz_l = sz - int(probe_d / mesh_length / 2)
sz_r = sz + int(probe_d / mesh_length / 2)

t_max = 4 * x_length / cl / dt

# ---------------- きずパラメータ確認 ----------------
print(f"f_pitch = {f_pitch}")
print(f"f_depth = {f_depth}")
print(f"Step Size = {step_size} mesh(es)")

# ---------------- FDTD配列準備 ----------------
T1 = cp.zeros((nx + 1, nz), dtype=float)
T3 = cp.zeros((nx + 1, nz), dtype=float)
T5 = cp.zeros((nx, nz + 1), dtype=float)
Ux = cp.zeros((nx, nz), dtype=float)
Uz = cp.zeros((nx + 1, nz + 1), dtype=float)

wave = np.zeros(int(t_max))

dtx = dt / dx
dtz = dt / dz

# 初期化
T13_isfree_np, T5_isfree_np = isfree_symmetric(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count_np, Uz_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

# GPUへ転送
T13_isfree = cp.asarray(T13_isfree_np)
T5_isfree = cp.asarray(T5_isfree_np)
Ux_free_count = cp.asarray(Ux_free_count_np)
Uz_free_count = cp.asarray(Uz_free_count_np)

start_time = time.time()

# ---------------- メイン時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0:
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    # (1) 静的境界
    T5[0:nx, 0] = 0
    T5[0:nx, nz] = 0
    T3[0, 0:nz] = 0
    T1[nx, 0:nz] = 0
    T3[nx, 0:nz] = 0
    T1[0, 0] = 0
    T3[0, 0] = 0
    T5[0, 0] = 0

    # (2) Uz 更新
    Uz[1:nx, 0]  = Uz[1:nx, 0]  - (4 / rho) * dtx * (T3[1:nx, 0])
    Uz[1:nx, nz] = Uz[1:nx, nz] - (4 / rho) * dtx * (-1 * T3[1:nx, nz - 1])
    Uz[nx, 1:nz] = Uz[nx, 1:nz] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:nz])

    # (3) 応力更新
    T1[1:nx, 0:nz] = T1[1:nx, 0:nz] - dtx * (
        (c11 * (Ux[1:nx, 0:nz] - Ux[0:nx - 1, 0:nz])) +
        (c13 * (Uz[1:nx, 1:nz + 1] - Uz[1:nx, 0:nz]))
    )
    T3[1:nx, 0:nz] = T3[1:nx, 0:nz] - dtx * (
        (c13 * (Ux[1:nx, 0:nz] - Ux[0:nx - 1, 0:nz])) +
        (c11 * (Uz[1:nx, 1:nz + 1] - Uz[1:nx, 0:nz]))
    )
    T5[0:nx, 1:nz] = T5[0:nx, 1:nz] - dtx * c55 * (
        Ux[0:nx, 1:nz] - Ux[0:nx, 0:nz - 1] +
        Uz[1:nx + 1, 1:nz] - Uz[0:nx, 1:nz]
    )

    # (4) 自由境界（きず）適用
    T1[T13_isfree == 0] = 0.0
    T3[T13_isfree == 0] = 0.0
    T5[T5_isfree[0:nx, :] == 0] = 0.0

    # (5) 音源
    if t < int(len(wave4)):
        T1[0, sz_l:sz_r] = wave4[t]
    else:
        Uz[0, sz_l:sz_r] = 0
        Ux[0, sz_l:sz_r] = 0
        T1[0, 0:nz] = 0
        T5[0, 0:nz] = 0

    # (6) 粒子速度更新
    Ux[0:nx, 0:nz] = cp.where(
        Ux_free_count[0:nx, 0:nz] < 4,
        Ux[0:nx, 0:nz] - (4 / rho / (4 - Ux_free_count[0:nx, 0:nz])) * dtx * (
            T1[1:nx + 1, 0:nz] - T1[0:nx, 0:nz] +
            T5[0:nx, 1:nz + 1] - T5[0:nx, 0:nz]
        ),
        0
    )
    Uz[1:nx, 1:nz] = cp.where(
        Uz_free_count[1:nx, 1:nz] < 4,
        Uz[1:nx, 1:nz] - (4 / rho / (4 - Uz_free_count[1:nx, 1:nz])) * dtz * (
            T3[1:nx, 1:nz] - T3[1:nx, 0:nz - 1] +
            T5[1:nx, 1:nz] - T5[0:nx - 1, 1:nz]
        ),
        0
    )

    cp.cuda.Device().synchronize()

    # (7) 記録
    if t > 0:
        wave[t] = cp.mean(T1[1, sz_l:sz_r])

wave = cp.asnumpy(wave)
end_time = time.time()

print(f"実行時間: {end_time - start_time:.2f} 秒")

# ---------------- CSV保存 ----------------
os.makedirs(output_dir, exist_ok=True)

csv_filename = f"sankaku_cupy_pitch{int(f_pitch * 1e5)}_depth{int(f_depth * 1e5)}.csv"

# ファイル名に step_size を追加
# csv_filename = f"sankaku_cupy_pitch{int(f_pitch * 1e5)}_depth{int(f_depth * 1e5)}_step{step_size}.csv"

csv_path = os.path.join(output_dir, csv_filename)
np.savetxt(csv_path, wave, delimiter=',')
print(f"保存しました → {csv_path}")
