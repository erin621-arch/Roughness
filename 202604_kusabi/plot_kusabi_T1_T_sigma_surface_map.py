import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
import os
import japanize_matplotlib

# ====== 読み込むファイルを指定 ======
input_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"
f_pitch = 2.00e-3
f_depth = 0.20e-3

probe_mode = "edge"   # "edge" または "center" を指定

npz_path_T1 = os.path.join(
    input_dir,
    f"kusabi_surface_map_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

npz_path_sigma = os.path.join(
    input_dir,
    f"kusabi_surface_sigma_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

probe_label = {
    "edge":   "(きずのピッチの端に探触子を配置)",
    "center": "(きずのピッチの中心に探触子を配置)",
}[probe_mode]

# 保存先のディレクトリを指定
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

# ====== データ読み込み（T1 用） ======
d1 = np.load(npz_path_T1)
T1_xslice   = d1['T1_xslice']
t_rec_start = int(d1['t_rec_start'])
t_rec_len   = int(d1['t_rec_len'])
dt          = float(d1['dt'])
mesh_length = float(d1['mesh_length'])
z_obs       = int(d1['z_obs'])
x_obs_start = int(d1['x_obs_start'])
x_obs_end   = int(d1['x_obs_end'])
f_pitch     = float(d1['f_pitch'])
f_depth     = float(d1['f_depth'])

# ====== データ読み込み（σ 用） ======
d2 = np.load(npz_path_sigma)
sigma_slope = d2['sigma_slope']
slope_z     = d2['slope_z']

# ====== 軸の構築 ======
t_axis         = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6   # [µs]
x_obs_start_mm = x_obs_start * mesh_length * 1e3
x_obs_end_mm   = x_obs_end   * mesh_length * 1e3
z_obs_mm       = z_obs       * mesh_length * 1e3
slope_z_mm     = slope_z     * mesh_length * 1e3                    # 斜面 z 座標 [mm]

# ====== 描画 ======
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(
    f'くさび形 {probe_label}\n'
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
    interpolation='bilinear'
)
ax.set_ylabel('x 方向 [mm]')
ax.set_title(f'T1 (z = {z_obs_mm:.2f} mm における縦方向応力)')
plt.colorbar(im1, ax=ax, label='[Pa]')
x_ticks = np.arange(x_obs_start_mm, x_obs_end_mm + 0.001, 0.05)
ax.set_yticks(x_ticks)
ax.set_yticklabels([f"{v:.2f}" for v in x_ticks])

# ---- σ = sqrt(T1² + T3²): 斜面上の合力マップ ----
ax = axes[1]
_vmax_sigma = np.percentile(np.abs(sigma_slope), 98)
im2 = ax.imshow(
    sigma_slope,
    aspect='auto',
    cmap='bwr',
    vmin=-_vmax_sigma, vmax=_vmax_sigma,
    extent=[t_axis[0], t_axis[-1], slope_z_mm[-1], slope_z_mm[0]],
    interpolation='nearest'
)
ax.set_ylabel('z 方向 [mm]')
ax.set_title(r'$\sigma_{tt} = T_1\sin^2\theta+2T_5\sin\theta\cos\theta+T_3\cos^2\theta$')
plt.colorbar(im2, ax=ax, label='[Pa]')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]')

plt.tight_layout()

fig_name = os.path.join(
    output_dir,
    f"kusabi_T1_sigma_surface_map_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(fig_name, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_name}")
plt.show()
