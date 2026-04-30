import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import japanize_matplotlib

# ====== 読み込むファイルを指定 ======
input_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"
f_pitch = 2.00e-3
f_depth = 0.20e-3

npz_path = os.path.join(
    input_dir,
    f"ushape_surface_map_center_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

# ====== データ読み込み ======
d = np.load(npz_path)
T1_corners_r = d['T1_corners_r']   # (n_points, t_rec_len)
T3_gap       = d['T3_gap']         # (gap_width, t_rec_len)
corner_x_r   = d['corner_x_r']
corner_z_r   = d['corner_z_r']
t_rec_start  = int(d['t_rec_start'])
t_rec_len    = int(d['t_rec_len'])
dt           = float(d['dt'])
mesh_length  = float(d['mesh_length'])
f_pitch      = float(d['f_pitch'])
f_depth      = float(d['f_depth'])
gap_z_start  = int(d['gap_z_start'])
gap_z_end    = int(d['gap_z_end'])

# ====== 軸の構築 ======
t_axis     = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6   # [µs]
n_pts      = T1_corners_r.shape[0]
gap_z_axis = np.arange(gap_z_start, gap_z_end) * mesh_length * 1e3  # [mm]

# ====== 描画 ======
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(
    f'U字型（円形）の 表面の応力について\n'
    f'(pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.3f} mm)',
    fontsize=12
)

# ---- T1: 各コーナー計測点 ----
ax = axes[0]
_vmax_T1 = np.percentile(np.abs(T1_corners_r), 98)
im1 = ax.imshow(
    T1_corners_r,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax_T1, vmax=_vmax_T1,
    extent=[t_axis[0], t_axis[-1], n_pts + 0.5, 0.5],
    interpolation='nearest'
)
ax.set_ylabel('計測点')
ax.set_title('T1（各コーナー計測点）')
ax.set_yticks(np.arange(1, n_pts + 1))
ax.set_yticklabels(
    [f"Pt{i+1} (x={corner_x_r[i]}, z={corner_z_r[i]})" for i in range(n_pts)],
    fontsize=8
)
plt.colorbar(im1, ax=ax, label='[Pa]')

# ---- T3: ギャップ連続記録 ----
ax = axes[1]
_vmax_T3 = np.percentile(np.abs(T3_gap), 98)
im2 = ax.imshow(
    T3_gap,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax_T3, vmax=_vmax_T3,
    extent=[t_axis[0], t_axis[-1], gap_z_axis[-1], gap_z_axis[0]],
    interpolation='bilinear'
)
ax.set_ylabel('z 方向 [mm]')
ax.set_title('T3（すきま部分のみ）')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
plt.colorbar(im2, ax=ax, label='[Pa]')

axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]')

plt.tight_layout()

fig_name = os.path.join(
    output_dir,
    f"ushape_T1_T3_surface_map_center_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(fig_name, dpi=150, bbox_inches='tight')
print(f"saved: {fig_name}")
plt.show()
