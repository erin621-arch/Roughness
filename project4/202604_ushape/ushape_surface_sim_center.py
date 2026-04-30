import numpy as np
import cupy as cp
import os
import time

# ================== 調整パラメータ ==================

output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

f_width   = 0.25e-3    # 幅 w [m]（固定）
f_depth   = 0.20e-3    # 全深さ d [m]
f_pitch   = 2.00e-3    # ピッチ P = W + Gap [m]
step_size = 1

# ===================================================

x_length    = 0.02
z_length    = 0.04
mesh_length = 1.0e-5

nx = int(round(x_length / mesh_length))
nz = int(round(z_length / mesh_length))

dx = x_length / nx
dz = z_length / nz

rho = 7840
E   = 206e9
G   = 80e9
V   = 0.27

cl  = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt    = dx / cl / np.sqrt(6)
f_hz  = 4.7e6
n_cyc = (1 / f_hz) / dt

t_max = 4 * x_length / cl / dt
dtx   = dt / dx
dtz   = dt / dz

# ---------------- U字型マスク生成 ----------------
def isfree_u_shape(nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size):
    T13_isfree = np.ones((nx + 1, nz))
    T5_isfree  = np.ones((nx, nz + 1))

    mn_w = int(round(f_width / mesh_length))
    if mn_w % 2 == 0:
        mn_w -= 1
    mn_d        = int(round(f_depth / mesh_length))
    mn_r        = mn_w // 2
    mn_straight = mn_d - mn_r
    mn_p_val    = max(1, int(round(f_pitch / mesh_length)))
    mn_nf       = max(0, mn_p_val - mn_w)
    mn_period   = mn_w + mn_nf

    T13_isfree[0,  0:nz] = 0
    T13_isfree[nx, 0:nz] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, nz]  = 0

    num_f = int(np.ceil(nz / mn_period)) + 1
    for i in range(num_f):
        z_start  = i * mn_period
        if z_start >= nz:
            break
        z_end    = min(z_start + mn_w, nz)
        z_center = (z_start + z_end) // 2

        for d in range(mn_d):
            xi     = (nx - 1) - d
            if xi < 0:
                break
            d_step = (d // step_size) * step_size
            if d_step < mn_straight:
                width_at_d = mn_w
            else:
                arc_d = d_step - mn_straight
                if arc_d < mn_r:
                    width_at_d = int(np.sqrt(mn_r**2 - arc_d**2) * 2)
                else:
                    width_at_d = 0
            if width_at_d > 0 and width_at_d % 2 == 0:
                width_at_d -= 1
            if width_at_d < 0:
                width_at_d = 0

            half = width_at_d // 2
            zl   = max(z_center - half, 0)
            zr   = min(z_center + half + 1, nz)
            if zl < zr:
                T5_isfree[xi, zl:zr + 1] = 0
                if xi < nx + 1:
                    T13_isfree[xi, zl:zr] = 0
                if xi + 1 < nx + 1:
                    T13_isfree[xi + 1, zl:zr] = 0

    void_adj = np.zeros((nx, nz + 1), dtype=bool)
    void_adj[:, 1:]  |= (T13_isfree[:nx, :] == 0)
    void_adj[:, :nz] |= (T13_isfree[:nx, :] == 0)
    T5_isfree[void_adj] = 0

    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 ----------------
def around_free(T13_isfree, T5_isfree):
    Ux_free_count = np.zeros((nx, nz), dtype=float)
    Uz_free_count = np.zeros((nx + 1, nz + 1), dtype=float)

    for i in range(nx):
        for j in range(nz):
            if T13_isfree[i,     j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i,  j]     == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i,  j + 1] == 0: Ux_free_count[i, j] += 1

    for i in range(nx + 1):
        for j in range(nz + 1):
            if j == 0 or j == nz or i == 0 or i == nx:
                Uz_free_count[i, j] += 1
            if j > 0  and i < nx + 1 and T13_isfree[i, j - 1] == 0: Uz_free_count[i, j] += 1
            if j < nz and i < nx + 1 and T13_isfree[i, j]     == 0: Uz_free_count[i, j] += 1
            if i > 0  and j < nz + 1 and T5_isfree[i - 1, j]  == 0: Uz_free_count[i, j] += 1
            if i < nx and j < nz + 1 and T5_isfree[i, j]       == 0: Uz_free_count[i, j] += 1

    dir1 = np.ones((nx + 1, nz + 1), dtype=bool); dir1[:, 1:]  = (T13_isfree == 0)
    dir2 = np.ones((nx + 1, nz + 1), dtype=bool); dir2[:, :nz] = (T13_isfree == 0)
    dir3 = np.ones((nx + 1, nz + 1), dtype=bool); dir3[1:, :]  = (T5_isfree  == 0)
    dir4 = np.ones((nx + 1, nz + 1), dtype=bool); dir4[:nx, :] = (T5_isfree  == 0)
    Uz_free_count[dir1 & dir2 & dir3 & dir4] = 4

    return Ux_free_count, Uz_free_count

# ---------------- 入射波形 ----------------
wn    = 2.5
wave4 = np.zeros(int(wn * n_cyc), dtype=float)
for ms in range(len(wave4)):
    wave4[ms] = ((1 - np.cos(2 * np.pi * f_hz * dt * ms / wn)) / 2
                 * np.sin(2 * np.pi * f_hz * dt * ms))

# ---------------- コーナー計算（_depth_from_center は mn_* グローバルを参照） ----------------
def _depth_from_center(dz_abs):
    max_d = 0
    for d in range(mn_d):
        d_step = (d // step_size) * step_size
        if d_step < mn_straight:
            width_at_d = mn_w
        else:
            arc_d = d_step - mn_straight
            if arc_d < mn_r:
                width_at_d = int(np.sqrt(mn_r**2 - arc_d**2) * 2)
            else:
                width_at_d = 0
            if width_at_d > 0 and width_at_d % 2 == 0:
                width_at_d -= 1
            if width_at_d < 0:
                width_at_d = 0
        if dz_abs <= width_at_d // 2:
            max_d = d + 1
    return max_d

def groove_corners(zs, ze, zc, keep_side):
    czs, cxs = [], []
    depths = {z: _depth_from_center(abs(z - zc)) for z in range(zs - 1, ze + 2)}
    for z in range(zs - 1, ze):
        d_cur  = depths.get(z,     0)
        d_next = depths.get(z + 1, 0)
        if d_cur == d_next:
            continue
        z_pos = z + 1
        if keep_side == 'right' and z_pos <= zc: continue
        if keep_side == 'left'  and z_pos >  zc: continue
        if d_cur == 0:
            czs.append(z_pos); cxs.append(nx)
            czs.append(z_pos); cxs.append(nx - d_next)
        elif d_next == 0:
            czs.append(z_pos); cxs.append(nx)
            czs.append(z_pos); cxs.append(nx - d_cur)
        else:
            czs.append(z_pos); cxs.append(nx - max(d_cur, d_next))
    return czs, cxs

os.makedirs(output_dir, exist_ok=True)

print(f"Pitch={f_pitch*1e3:.2f} mm  Width={f_width*1e3:.2f} mm  Depth={f_depth*1e3:.3f} mm")

# ---- ジオメトリ ----
mn_w = int(round(f_width / mesh_length))
if mn_w % 2 == 0:
    mn_w -= 1
mn_d        = int(round(f_depth / mesh_length))
mn_r        = mn_w // 2
mn_straight = mn_d - mn_r
mn_p_val    = int(round(f_pitch / mesh_length))
mn_nf       = max(0, mn_p_val - mn_w)
mn_period   = mn_w + mn_nf

# 溝1・溝2の位置（nz/2 付近の溝を基準）
_sz_ref         = int(nz / 2)
i_near          = max(0, _sz_ref // mn_period)
z_groove_start  = i_near * mn_period
z_groove_end    = z_groove_start + mn_w
z_groove_center = (z_groove_start + z_groove_end) // 2
z2_start        = z_groove_start + mn_period
z2_end          = z2_start + mn_w
z2_center       = (z2_start + z2_end) // 2

# 探触子位置（center モード: ギャップ中心）
sz      = (z_groove_end + z2_start) // 2
probe_d = 0.007
sz_l    = sz - int(probe_d / mesh_length / 2)
sz_r    = sz + int(probe_d / mesh_length / 2)

print(f"溝1: z=[{z_groove_start}, {z_groove_end})  溝2: z=[{z2_start}, {z2_end})")
print(f"ギャップ: [{z_groove_end*mesh_length*1e3:.2f}, {z2_start*mesh_length*1e3:.2f}) mm")
print(f"探触子中心: z={sz} ({sz*mesh_length*1e3:.3f} mm)")

# ---- T1 コーナー点（溝1 右側のみ）----
cz1, cx1   = groove_corners(z_groove_start, z_groove_end, z_groove_center, 'right')
corner_z_r = np.array(cz1, dtype=int);  corner_x_r = np.array(cx1, dtype=int)

valid_r    = (corner_x_r < nx) & (corner_x_r >= 0) & (corner_z_r >= 0) & (corner_z_r < nz)
corner_z_r = corner_z_r[valid_r];  corner_x_r = corner_x_r[valid_r]

# Point 9: x=nx-1=1999 を末尾に追加（境界x=2000の代替）
corner_x_r = np.append(corner_x_r, nx - 1)
corner_z_r = np.append(corner_z_r, z_groove_end)
print(f"T1 計測点数 (右コーナー): {len(corner_z_r)}")

# ---- T3 計測範囲 ----
_gap_z_start = z_groove_end
_gap_z_end   = z2_start
_gap_x       = nx - 1
_gap_width   = _gap_z_end - _gap_z_start
print(f"T3 連続記録: z=[{_gap_z_start*mesh_length*1e3:.2f}, {_gap_z_end*mesh_length*1e3:.2f}) mm  ({_gap_width} 点)")

# ---- 記録バッファ ----
_t_rec_start  = 4000
_t_rec_len    = 5000
_T1_corners_r = np.zeros((len(corner_z_r), _t_rec_len), dtype=float)
_T3_gap       = np.zeros((_gap_width,      _t_rec_len), dtype=float)

# ---- GPU 配列確保 ----
T1 = cp.zeros((nx + 1, nz),     dtype=float)
T3 = cp.zeros((nx + 1, nz),     dtype=float)
T5 = cp.zeros((nx,     nz + 1), dtype=float)
Ux = cp.zeros((nx,     nz),     dtype=float)
Uz = cp.zeros((nx + 1, nz + 1), dtype=float)
wave_recv = np.zeros(int(t_max))

T13_isfree_np, T5_isfree_np = isfree_u_shape(
    nx, nz, f_width, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count_np, Uz_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

T13_isfree     = cp.asarray(T13_isfree_np)
T5_isfree_gpu  = cp.asarray(T5_isfree_np)
Ux_free_count  = cp.asarray(Ux_free_count_np)
Uz_free_count  = cp.asarray(Uz_free_count_np)
_corner_x_r_cp = cp.array(corner_x_r)
_corner_z_r_cp = cp.array(corner_z_r)

print(f"\nt_max = {int(t_max)},  dt = {dt:.3e} s")
start_time = time.time()

# ---------------- 時間ループ ----------------
for t in range(int(t_max)):
    if t % 500 == 0:
        print(f"  {t}/{int(t_max)} ({t / t_max:.1%})")

    T5[0:nx, 0] = 0;  T5[0:nx, nz] = 0
    T3[0, 0:nz] = 0;  T3[nx, 0:nz] = 0
    T1[nx, 0:nz] = 0; T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

    Uz[1:nx, 0]  -= (4 / rho) * dtx * T3[1:nx, 0]
    Uz[1:nx, nz] -= (4 / rho) * dtx * (-T3[1:nx, nz - 1])
    Uz[nx, 1:nz] -= (4 / rho) * dtx * (-T5[nx - 1, 1:nz])

    T1[1:nx, :] -= dtx * (c11 * (Ux[1:nx, :] - Ux[0:nx - 1, :])
                          + c13 * (Uz[1:nx, 1:] - Uz[1:nx, :-1]))
    T3[1:nx, :] -= dtx * (c13 * (Ux[1:nx, :] - Ux[0:nx - 1, :])
                          + c11 * (Uz[1:nx, 1:] - Uz[1:nx, :-1]))
    T5[:, 1:nz] -= dtx * c55 * (Ux[:, 1:] - Ux[:, :-1]
                                 + Uz[1:, 1:nz] - Uz[:-1, 1:nz])

    T1[T13_isfree == 0] = 0.0
    T3[T13_isfree == 0] = 0.0
    T5[T5_isfree_gpu[0:nx, :] == 0] = 0.0

    if t < len(wave4):
        T1[0, sz_l:sz_r] = wave4[t]
    else:
        Uz[0, sz_l:sz_r] = 0; Ux[0, sz_l:sz_r] = 0
        T1[0, 0:nz] = 0;      T5[0, 0:nz] = 0

    Ux[0:nx, 0:nz] = cp.where(
        Ux_free_count < 4,
        Ux - (4 / rho / (4 - Ux_free_count)) * dtx * (
            T1[1:nx + 1, :] - T1[0:nx, :] + T5[:, 1:nz + 1] - T5[:, 0:nz]
        ), 0
    )
    Uz[1:nx, 1:nz] = cp.where(
        Uz_free_count[1:nx, 1:nz] < 4,
        Uz[1:nx, 1:nz] - (4 / rho / (4 - Uz_free_count[1:nx, 1:nz])) * dtz * (
            T3[1:nx, 1:nz] - T3[1:nx, :-1] + T5[1:nx, 1:nz] - T5[:-1, 1:nz]
        ), 0
    )

    if t > 0:
        wave_recv[t] = float(cp.mean(T1[1, sz_l:sz_r]))

    if _t_rec_start <= t < _t_rec_start + _t_rec_len:
        ti = t - _t_rec_start
        _T1_corners_r[:, ti] = cp.asnumpy(T1[_corner_x_r_cp, _corner_z_r_cp])
        _T3_gap[:, ti]       = cp.asnumpy(T3[_gap_x, _gap_z_start:_gap_z_end])

    if t % 1000 == 0:
        cp.cuda.Device().synchronize()

end_time = time.time()
print(f"完了. 計算時間: {end_time - start_time:.2f} s")

# ---------------- 保存 ----------------
npz_path = os.path.join(
    output_dir,
    f"ushape_surface_map_center_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)
np.savez(
    npz_path,
    T1_corners_r = _T1_corners_r,
    T3_gap       = _T3_gap,
    corner_x_r   = corner_x_r,
    corner_z_r   = corner_z_r,
    gap_x        = np.array(_gap_x),
    gap_z_start  = np.array(_gap_z_start),
    gap_z_end    = np.array(_gap_z_end),
    t_rec_start  = np.array(_t_rec_start),
    t_rec_len    = np.array(_t_rec_len),
    dt           = np.array(dt),
    mesh_length  = np.array(mesh_length),
    f_pitch      = np.array(f_pitch),
    f_width      = np.array(f_width),
    f_depth      = np.array(f_depth),
    mn_d         = np.array(mn_d),
    mn_w         = np.array(mn_w),
    z_groove_end = np.array(z_groove_end),
    z2_start     = np.array(z2_start),
)
print(f"保存: {npz_path}")
