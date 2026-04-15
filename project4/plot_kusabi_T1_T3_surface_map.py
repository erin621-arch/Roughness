import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
import os
import japanize_matplotlib

# ====== 読み込むファイルを指定 ======
input_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"
f_pitch = 1.25e-3
f_depth = 0.20e-3
npz_path = os.path.join(
    input_dir,
    f"kusabi_surface_map_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

# 保存先のディレクトリを指定
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

# ====== データ読み込み ======
d = np.load(npz_path)
T1_xslice   = d['T1_xslice']
T3_surface  = d['T3_surface']
t_rec_start = int(d['t_rec_start'])
t_rec_len   = int(d['t_rec_len'])
dt          = float(d['dt'])
mesh_length = float(d['mesh_length'])
z_obs       = int(d['z_obs'])
x_obs_start = int(d['x_obs_start'])
x_obs_end   = int(d['x_obs_end'])
meas_z      = d['meas_z']
f_pitch     = float(d['f_pitch'])
f_depth     = float(d['f_depth'])
mn_d        = int(d['mn_d'])
z_rec_start = int(d['z_rec_start'])

# ====== 軸の構築 ======
t_axis      = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6   # [µs]
z_axis_mm   = meas_z * mesh_length * 1e3                          # 凸角 z 座標 [mm]
x_obs_start_mm = x_obs_start * mesh_length * 1e3                  # [mm]
x_obs_end_mm   = x_obs_end   * mesh_length * 1e3                  # [mm]
z_obs_mm       = z_obs       * mesh_length * 1e3                  # [mm]
z_rec_start_mm = z_rec_start * mesh_length * 1e3                  # [mm]

# ====== 描画 ======
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(
    f'くさび形\n'
    f'(pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.2f} mm)',
    fontsize=12
)

# ---- T1: z=z_obs における x 方向断面マップ ----
ax = axes[0]
_vmax_T1 = np.percentile(np.abs(T1_xslice), 98)
im1 = ax.imshow(
    T1_xslice,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax_T1, vmax=_vmax_T1,
    extent=[t_axis[0], t_axis[-1], x_obs_end_mm, x_obs_start_mm],
    interpolation='nearest'
)
ax.set_ylabel('x 方向 [mm]')
ax.set_title(f'T1 (z = 21.24mm における縦方向応力)')
plt.colorbar(im1, ax=ax, label='[Pa]')
x_ticks = np.arange(x_obs_start_mm, x_obs_end_mm + 0.001, 0.05)
ax.set_yticks(x_ticks)
ax.set_yticklabels([f"{v:.2f}" for v in x_ticks])

# ---- T3: 凸角 20点の表面マップ ----
ax = axes[1]
_vmax_T3 = np.percentile(np.abs(T3_surface), 98)
im2 = ax.imshow(
    T3_surface,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax_T3, vmax=_vmax_T3,
    extent=[t_axis[0], t_axis[-1], z_axis_mm[-1], z_axis_mm[0]],
    interpolation='nearest'
)
ax.set_ylabel('z 方向 [mm]')
ax.set_title(f'T3 (横方向応力)')
plt.colorbar(im2, ax=ax, label='[Pa]')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]')

plt.tight_layout()

fig_name = os.path.join(
    output_dir,
    f"kusabi_T1_T3_surface_map_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)

plt.savefig(fig_name, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_name}")
plt.show()
