import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import matplotlib
import time

# ================== 調整パラメータ ==================

output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

# ★くさび形(IMG_2910) のパラメータ
f_pitch = 1.25e-3   # ピッチ p [m]
f_depth = 0.20e-3   # 深さ d [m]

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

# ---------------- くさび形マスク生成 ----------------
def isfree_kusabi(nx, nz, f_pitch, f_depth, mesh_length, step_size):
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

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
                T13_isfree[cut_top : nx, z] = 0
                T5_isfree[cut_top : nx, z] = 0
                T5_isfree[cut_top : nx, z+1] = 0

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
            if j > 0 and i < nx + 1:
                if T13_isfree[i, j - 1] == 0: Uz_free_count[i, j] += 1
            if j < nz and i < nx + 1:
                if T13_isfree[i, j] == 0:     Uz_free_count[i, j] += 1
            if i > 0 and j < nz + 1:
                if T5_isfree[i - 1, j] == 0:  Uz_free_count[i, j] += 1
            if i < nx and j < nz + 1:
                if T5_isfree[i, j] == 0:      Uz_free_count[i, j] += 1

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

mn_p = int(round(f_pitch / mesh_length))   # 125
mn_d = int(round(f_depth / mesh_length))   # 20

# ★ T1 計測点: z=2125 の列全体 (x: 0 ~ nx)
_z_meas = (sz // mn_p + 1) * mn_p   # = 2125

print(f"T1計測z座標 (グリッド): {_z_meas}  ({_z_meas * mesh_length * 1e3:.3f} mm)")

_x_meas = np.arange(0, nx + 1, dtype=int)   # shape=(nx+1,) = (2001,)

# ★ T3 計測点: kusabi_surface_sim.py と同じ（外側の凸角 + 底面点）
_pitch_i     = sz // mn_p
_z_rec_start = _pitch_i * mn_p         # = 2000
_z_rec_end   = (_pitch_i + 1) * mn_p  # = 2125

_z_all   = np.arange(_z_rec_start, _z_rec_end + 1)
_dep_all = np.array([
    (int(mn_d * (1.0 - (z - (z // mn_p) * mn_p) / mn_p)) // step_size) * step_size
    for z in _z_all
], dtype=int)
_x_all = nx - _dep_all

_corner_z = []
_corner_x = []
for i in range(len(_z_all) - 1):
    if _dep_all[i+1] < _dep_all[i]:
        _corner_z.append(int(_z_all[i+1]))
        _corner_x.append(int(_x_all[i]))

_corner_z.append(_z_rec_end)
_corner_x.append(nx)

_meas_z = np.array(_corner_z, dtype=int)
_meas_x = np.array(_corner_x, dtype=int)

_z_surf_cp = cp.array(_meas_z)
_x_surf_cp = cp.array(_meas_x)

# 記録時間帯
_t_rec_start_t = 4000
_t_rec_len     = 5000
_T1_map = np.zeros((nx + 1,       _t_rec_len), dtype=float)  # T1: 全x × 時間
_T3_map = np.zeros((len(_meas_z), _t_rec_len), dtype=float)  # T3: 表面点 × 時間

# ---------------- 実行準備 ----------------
print(f"Pitch(p) = {f_pitch*1000} mm")
print(f"Depth(d) = {f_depth*1000} mm")
print(f"Step Size = {step_size} mesh(es)")
print(f"nx={nx}, nz={nz}, t_max={int(t_max)}")

T1 = cp.zeros((nx + 1, nz), dtype=float)
T3 = cp.zeros((nx + 1, nz), dtype=float)
T5 = cp.zeros((nx, nz + 1), dtype=float)
Ux = cp.zeros((nx, nz), dtype=float)
Uz = cp.zeros((nx + 1, nz + 1), dtype=float)
wave = np.zeros(int(t_max))

dtx = dt / dx
dtz = dt / dz

T13_isfree_np, T5_isfree_np = isfree_kusabi(nx, nz, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count_np, Uz_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

T13_isfree = cp.asarray(T13_isfree_np)
T5_isfree = cp.asarray(T5_isfree_np)
Ux_free_count = cp.asarray(Ux_free_count_np)
Uz_free_count = cp.asarray(Uz_free_count_np)

start_time = time.time()

# ---------------- 時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0:
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    T5[0:nx, 0] = 0; T5[0:nx, nz] = 0
    T3[0, 0:nz] = 0; T3[nx, 0:nz] = 0
    T1[nx, 0:nz] = 0; T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

    Uz[1:nx, 0]  -= (4/rho)*dtx * T3[1:nx, 0]
    Uz[1:nx, nz] -= (4/rho)*dtx * (-T3[1:nx, nz-1])
    Uz[nx, 1:nz] -= (4/rho)*dtx * (-T5[nx-1, 1:nz])

    T1[1:nx, :] -= dtx * (c11*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c13*(Uz[1:nx,1:] - Uz[1:nx,:-1]))
    T3[1:nx, :] -= dtx * (c13*(Ux[1:nx,:] - Ux[0:nx-1,:]) + c11*(Uz[1:nx,1:] - Uz[1:nx,:-1]))
    T5[:, 1:nz] -= dtx * c55 * (Ux[:,1:] - Ux[:,:-1] + Uz[1:,1:nz] - Uz[:-1,1:nz])

    T1[T13_isfree == 0] = 0.0
    T3[T13_isfree == 0] = 0.0
    T5[T5_isfree[0:nx, :] == 0] = 0.0

    if t < int(len(wave4)):
        T1[0, sz_l:sz_r] = wave4[t]
    else:
        Uz[0, sz_l:sz_r] = 0; Ux[0, sz_l:sz_r] = 0
        T1[0, 0:nz] = 0; T5[0, 0:nz] = 0

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

    # ★ T1: z=_z_meas 列全体(全x) / T3: 表面凸角+底面点
    if _t_rec_start_t <= t < _t_rec_start_t + _t_rec_len:
        ti = t - _t_rec_start_t
        _T1_map[:, ti] = cp.asnumpy(T1[:, _z_meas])
        _T3_map[:, ti] = cp.asnumpy(T3[_x_surf_cp, _z_surf_cp])

    if t % 1000 == 0:
        cp.cuda.Device().synchronize()

wave = cp.asnumpy(wave)
end_time = time.time()
print(f"Done. Time: {end_time - start_time:.2f} s")

# ---------------- 保存 ----------------
os.makedirs(output_dir, exist_ok=True)

_npy_T1 = os.path.join(
    output_dir,
    f"kusabi_T1_z{_z_meas}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npy"
)
_npy_T3 = os.path.join(
    output_dir,
    f"kusabi_T3_surface_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npy"
)
np.save(_npy_T1, _T1_map)
np.save(_npy_T3, _T3_map)
print(f"T1 map saved: {_npy_T1}")
print(f"T3 map saved: {_npy_T3}")

# ---------------- 描画 ----------------
matplotlib.rcParams['font.family'] = 'Noto Sans JP'

_t_axis    = (np.arange(_t_rec_len) + _t_rec_start_t) * dt * 1e6  # [µs]
_x_axis_mm = _x_meas * mesh_length * 1e3                            # x [mm]
_z_axis_mm = _meas_z * mesh_length * 1e3                            # z [mm]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    f'時空間マップ\n'
    f'pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.2f} mm',
    fontsize=12
)

# --- T1: 縦軸 = x ---
ax = axes[0]
_vmax = np.percentile(np.abs(_T1_map), 98)
im = ax.imshow(
    _T1_map,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax, vmax=_vmax,
    extent=[_t_axis[0], _t_axis[-1], _x_axis_mm[-1], _x_axis_mm[0]],
    interpolation='nearest'
)
ax.set_ylabel('x 方向位置 [mm]')
ax.set_title(f'T1 (縦方向応力)  @ z={_z_meas} ({_z_meas * mesh_length * 1e3:.3f} mm)')
plt.colorbar(im, ax=ax, label='T1 [Pa]')

# --- T3: 縦軸 = z (kusabi_surface_sim.py と同じ) ---
ax = axes[1]
_vmax = np.percentile(np.abs(_T3_map), 98)
im = ax.imshow(
    _T3_map,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax, vmax=_vmax,
    extent=[_t_axis[0], _t_axis[-1], _z_axis_mm[-1], _z_axis_mm[0]],
    interpolation='nearest'
)
ax.set_ylabel('z 方向位置 [mm]')
ax.set_title(f'T3 (横方向応力)  外側の凸角 + 底面点 計{len(_meas_z)}点')
plt.colorbar(im, ax=ax, label='T3 [Pa]')

axes[-1].set_xlabel('Time [µs]')

plt.tight_layout()
_fig_name = os.path.join(
    output_dir,
    f"kusabi_T1xaxis_T3zaxis_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(_fig_name, dpi=150, bbox_inches='tight')
print(f"Figure saved: {_fig_name}")
plt.show()

# CSV (受信波形)
csv_name = f"kusabi2_cupy_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.csv"
save_path = os.path.join(output_dir, csv_name)
np.savetxt(save_path, wave, delimiter=',')
print(f"Saved to: {save_path}")
