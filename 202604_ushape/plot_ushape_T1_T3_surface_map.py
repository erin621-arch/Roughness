import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import japanize_matplotlib

# ====== 読み込むファイルを指定 ======
input_dir  = r"C:/Users/cs16/Roughness/project4/tmp_output"
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

probe_mode = "center"   # "edge" または "center" を指定
f_pitch    = 2.00e-3
f_depth    = 0.20e-3

probe_label = {
    "edge":   "(溝の端に探触子を配置)",
    "center": "(溝と溝の隙間中心に探触子を配置)",
}[probe_mode]

# ======================================================

npz_path = os.path.join(
    input_dir,
    f"ushape_surface_map_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

# ====== データ読み込み ======
d = np.load(npz_path)
T1_corners_r = d['T1_corners_r']
T3_gap       = d['T3_gap']
corner_z_r   = d['corner_z_r']
gap_x        = int(d['gap_x'])
gap_z_start  = int(d['gap_z_start'])
gap_z_end    = int(d['gap_z_end'])
t_rec_start  = int(d['t_rec_start'])
t_rec_len    = int(d['t_rec_len'])
dt           = float(d['dt'])
mesh_length  = float(d['mesh_length'])
f_pitch_     = float(d['f_pitch'])
f_width_     = float(d['f_width'])
f_depth_     = float(d['f_depth'])

has_left = 'T1_corners_l' in d
if has_left:
    T1_corners_l = d['T1_corners_l']
    corner_z_l   = d['corner_z_l']

# ====== 軸の構築 ======
t_axis        = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6
corner_z_r_mm = corner_z_r * mesh_length * 1e3
gap_z_mm      = np.arange(gap_z_start, gap_z_end) * mesh_length * 1e3
if has_left:
    corner_z_l_mm = corner_z_l * mesh_length * 1e3

# ====== 描画 ======
n_plots = 3 if has_left else 2
fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
fig.suptitle(
    f'U字型 {probe_label}\n'
    f'(pitch={f_pitch_*1e3:.2f} mm,  width={f_width_*1e3:.2f} mm,  depth={f_depth_*1e3:.3f} mm)',
    fontsize=12
)

# T1 右コーナー（溝1 右壁）
ax = axes[0]
_vmax = np.percentile(np.abs(T1_corners_r), 98)
im = ax.imshow(
    T1_corners_r,
    aspect='auto', cmap='bwr', vmin=-_vmax, vmax=_vmax,
    extent=[t_axis[0], t_axis[-1], corner_z_r_mm[-1], corner_z_r_mm[0]],
    interpolation='nearest'
)
ax.set_ylabel('z 方向 [mm]')
ax.set_title('T1 右コーナー (溝1 右壁  σ_xx)')
plt.colorbar(im, ax=ax, label='[Pa]')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

# T1 左コーナー（溝2 左壁）― edge モードのみ
if has_left:
    ax = axes[1]
    _vmax = np.percentile(np.abs(T1_corners_l), 98)
    im = ax.imshow(
        T1_corners_l,
        aspect='auto', cmap='bwr', vmin=-_vmax, vmax=_vmax,
        extent=[t_axis[0], t_axis[-1], corner_z_l_mm[-1], corner_z_l_mm[0]],
        interpolation='nearest'
    )
    ax.set_ylabel('z 方向 [mm]')
    ax.set_title('T1 左コーナー (溝2 左壁  σ_xx)')
    plt.colorbar(im, ax=ax, label='[Pa]')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

# T3 ギャップ
ax = axes[-1]
_vmax = np.percentile(np.abs(T3_gap), 98)
im = ax.imshow(
    T3_gap,
    aspect='auto', cmap='bwr', vmin=-_vmax, vmax=_vmax,
    extent=[t_axis[0], t_axis[-1], gap_z_mm[-1], gap_z_mm[0]],
    interpolation='bilinear'
)
ax.set_ylabel('z 方向 [mm]')
ax.set_title(
    f'T3 (ギャップ表面の横方向応力  σ_zz,  x = {gap_x * mesh_length * 1e3:.3f} mm)'
)
plt.colorbar(im, ax=ax, label='[Pa]')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]')
plt.tight_layout()

fig_name = os.path.join(
    output_dir,
    f"ushape_T1_T3_surface_map_{probe_mode}_pitch{int(f_pitch_*1e5)}_depth{int(f_depth_*1e5)}.png"
)
plt.savefig(fig_name, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_name}")
plt.show()
