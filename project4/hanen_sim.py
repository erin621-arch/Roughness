import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time

# ================== 調整パラメータ (図面仕様) ==================

# 出力先
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"  # 研究室PC
# output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"   # 自宅PC

# ★図面のパラメータ (U字型 / 弾丸型)
# w = 0.25, R = 0.125, Straight = 0.075
f_width = 0.25e-3    # 幅 w [m] (固定)
f_depth = 0.20e-3    # 全深さ d [m]
f_pitch = 1.25e-3    # ピッチ (左端から次の右端までの距離)

# ★階段の高さ（メッシュ数）
# R部分の滑らかさを調整
step_size = 1

# =======================================================================

# ---------------- 基本パラメータ ----------------
x_length = 0.02   # [m]
z_length = 0.04   # [m]
mesh_length = 1.00e-5  # メッシュサイズ [m]

nx = int(x_length / mesh_length)
nz = int(z_length / mesh_length)

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

# ---------------- U字型(弾丸型)きず生成関数 ----------------
def isfree_u_shape(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size):
    # 1:固体 / 0:空洞
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

    # --- 1. 寸法の離散化 ---
    mn_w = int(round(f_width / mesh_length))
    # 対称性のため奇数に固定
    if mn_w % 2 == 0:
        mn_w -= 1

    mn_d = int(round(f_depth / mesh_length))
    mn_r = mn_w // 2
    mn_straight = mn_d - mn_r

    # --- 2. ピッチとすき間の計算 ---
    mn_p_val = max(1, int(round(f_pitch / mesh_length)))

    # ★定義: ピッチ = 左端〜次右端
    # 距離 P = 幅W + すきまGap + 幅W
    # よって、すきまGap = P - 2W
    mn_nf = max(0, mn_p_val - 2 * mn_w)

    # 配置周期 (Start-to-Start) = 幅 + すきま
    mn_period = mn_w + mn_nf

    # 外枠
    T13_isfree[0, 0:nz]  = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    # きずの数
    num_f = int(np.ceil(nz / mn_period))

    for i in range(num_f):
        # 中心位置
        z_start = i * mn_period
        if z_start >= nz: break

        z_end = min(z_start + mn_w, nz)
        z_center = (z_start + z_end) // 2

        # 深さ方向ループ
        for d in range(mn_d):
            xi = (nx - 1) - d
            if xi < 0: break

            d_step = (d // step_size) * step_size
            width_at_d = 0

            # (A) 直線部分
            if d_step < mn_straight:
                width_at_d = mn_w
            # (B) 半円部分
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

    # ===== T5境界補正 =====
    # T5_isfree[ix, iz] の4隣接T13に空洞が含まれる場合は空洞(0)に設定
    # T5[ix, iz] の周囲T13: T13[ix, iz-1], T13[ix+1, iz-1],
    #                        T13[ix, iz  ], T13[ix+1, iz  ]
    void_adj = np.zeros((nx, nz + 1), dtype=bool)
    void_adj[:, 1:]  |= (T13_isfree[:nx,    :] == 0)  # T13[ix,   iz-1]
    void_adj[:, 1:]  |= (T13_isfree[1:nx+1, :] == 0)  # T13[ix+1, iz-1]
    void_adj[:, :nz] |= (T13_isfree[:nx,    :] == 0)  # T13[ix,   iz  ]
    void_adj[:, :nz] |= (T13_isfree[1:nx+1, :] == 0)  # T13[ix+1, iz  ]
    T5_isfree[void_adj] = 0

    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 ----------------
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
            elif 0 < i < nx and 0 < j < nz:
                if T13_isfree[i, j - 1] == 0: Uz_free_count[i, j] += 1
                if T13_isfree[i, j] == 0:     Uz_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0:  Uz_free_count[i, j] += 1
                if T5_isfree[i, j] == 0:      Uz_free_count[i, j] += 1

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
print(f"Shape: U-Shape/Hanen (Straight={f_depth-f_width/2:.2e}, R={f_width/2:.2e})")
print(f"Pitch = {f_pitch*1000} mm (Left-to-Right Edge)")
print(f"Depth = {f_depth*1000} mm")
print(f"Step Size = {step_size}")

# 配列確保
T1 = cp.zeros((nx + 1, nz), dtype=float)
T3 = cp.zeros((nx + 1, nz), dtype=float)
T5 = cp.zeros((nx, nz + 1), dtype=float)
Ux = cp.zeros((nx, nz), dtype=float)
Uz = cp.zeros((nx + 1, nz + 1), dtype=float)
wave = np.zeros(int(t_max))

dtx = dt / dx
dtz = dt / dz

# ★U字型関数呼び出し
T13_isfree_np, T5_isfree_np = isfree_u_shape(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count_np, Uz_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

# GPU転送
T13_isfree = cp.asarray(T13_isfree_np)
T5_isfree = cp.asarray(T5_isfree_np)
Ux_free_count = cp.asarray(Ux_free_count_np)
Uz_free_count = cp.asarray(Uz_free_count_np)

start_time = time.time()

# ---------------- 時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0:
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    # 境界
    T5[0:nx, 0] = 0; T5[0:nx, nz] = 0
    T3[0, 0:nz] = 0; T3[nx, 0:nz] = 0
    T1[nx, 0:nz] = 0; T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

    # Uz更新
    Uz[1:nx, 0]  -= (4/rho)*dtx * T3[1:nx, 0]
    Uz[1:nx, nz] -= (4/rho)*dtx * (-T3[1:nx, nz-1])
    Uz[nx, 1:nz] -= (4/rho)*dtx * (-T5[nx-1, 1:nz])

    # 応力更新
    T1[1:nx, :] -= dtx * (c11*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c13*(Uz[1:nx,1:] - Uz[1:nx,:-1]))
    T3[1:nx, :] -= dtx * (c13*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c11*(Uz[1:nx,1:] - Uz[1:nx,:-1]))
    T5[:, 1:nz] -= dtx * c55 * (Ux[:,1:] - Ux[:,:-1] + Uz[1:,1:nz] - Uz[:-1,1:nz])

    # ★きず内部応力ゼロ
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

    if t % 1000 == 0:
         cp.cuda.Device().synchronize()

wave = cp.asnumpy(wave)
end_time = time.time()
print(f"Done. Time: {end_time - start_time:.2f} s")

# ---------------- 保存 & 表示 ----------------
os.makedirs(output_dir, exist_ok=True)

csv_name = f"hanen_cupy_pitch{f_pitch*1e5:g}_depth{f_depth*1e5:g}_step{step_size}.csv"
# csv_name = f"hanen_cupy_pitch{f_pitch*1e5:g}_depth{f_depth*1e5:g}.csv"

save_path = os.path.join(output_dir, csv_name)

np.savetxt(save_path, wave, delimiter=',')
print(f"Saved to: {save_path}")
