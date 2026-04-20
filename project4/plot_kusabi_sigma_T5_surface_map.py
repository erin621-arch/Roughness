import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import japanize_matplotlib

# ====== 読み込むファイルを指定 ======
input_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"
f_pitch = 2.00e-3
f_depth = 0.20e-3

probe_mode = "edge"   # "edge" または "center"

npz_path = os.path.join(
    input_dir,
    f"kusabi_surface_sigma_T5_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"

probe_label = {
    "edge":   "(きずのピッチの端に探触子を配置)",
    "center": "(きずのピッチの中心に探触子を配置)",
}[probe_mode]

# ====== データ読み込み ======
d = np.load(npz_path)
T1_slope    = d['T1_slope']
T3_slope    = d['T3_slope']
T5_slope    = d['T5_slope']
sigma_slope = d['sigma_slope']
slope_z     = d['slope_z']
t_rec_start = int(d['t_rec_start'])
t_rec_len   = int(d['t_rec_len'])
dt          = float(d['dt'])
mesh_length = float(d['mesh_length'])
f_pitch     = float(d['f_pitch'])
f_depth     = float(d['f_depth'])

# ====== 軸の構築 ======
t_axis    = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6   # [µs]
slope_z_mm = slope_z * mesh_length * 1e3                       # [mm]

extent = [t_axis[0], t_axis[-1], slope_z_mm[-1], slope_z_mm[0]]

# ====== 描画 ======
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle(
    f'くさび形 {probe_label}\n'
    f'(pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.2f} mm)　斜面上の応力',
    fontsize=12
)

panels = [
    (T1_slope,    r'$T_1 = \sigma_{xx}$（x方向垂直応力）'),
    (T3_slope,    r'$T_3 = \sigma_{zz}$（z方向垂直応力）'),
    (T5_slope,    r'$T_5 = \sigma_{xz}$（せん断応力）'),
    (sigma_slope, r'$\sigma = \mathrm{sign}(\sigma_{tt})\,\sqrt{T_1^2+T_3^2}$（符号付き合力）'),
]

for ax, (data, title) in zip(axes, panels):
    vmax = np.percentile(np.abs(data), 98)
    im = ax.imshow(
        data,
        aspect='auto',
        cmap='bwr',
        vmin=-vmax, vmax=vmax,
        extent=extent,
        interpolation='nearest'
    )
    ax.set_ylabel('z 方向 [mm]')
    ax.set_title(title)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.colorbar(im, ax=ax, label='[Pa]')

axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]')

plt.tight_layout()

fig_name = os.path.join(
    output_dir,
    f"kusabi_sigma_T5_surface_map_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(fig_name, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_name}")
plt.show()
