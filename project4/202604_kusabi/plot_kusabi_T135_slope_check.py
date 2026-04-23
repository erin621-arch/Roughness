import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import japanize_matplotlib

# ====== 読み込むファイルを指定 ======
input_dir  = r"C:/Users/cs16/Roughness/project4/tmp_output"
f_pitch    = 2.00e-3
f_depth    = 0.20e-3
probe_mode = "edge"   # "edge" または "center"

npz_path = os.path.join(
    input_dir,
    f"kusabi_surface_sigma_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.npz"
)

output_dir = input_dir

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
mn_p        = int(d['mn_p'])
mn_d        = int(d['mn_d'])

# ====== 軸の構築 ======
t_axis     = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6
slope_z_mm = slope_z * mesh_length * 1e3

# ====== σ_tt の係数と各成分の最大値を表示 ======
_L    = np.sqrt(mn_d**2 + mn_p**2)
sin_t = mn_d / _L
cos_t = mn_p / _L
print(f"θ = {np.degrees(np.arctan(mn_d/mn_p)):.2f}°")
print(f"sin²θ        = {sin_t**2:.4f}")
print(f"2sinθcosθ    = {2*sin_t*cos_t:.4f}")
print(f"cos²θ        = {cos_t**2:.4f}")
print()
print(f"T1 最大値: {np.max(np.abs(T1_slope)):.4f} Pa")
print(f"T3 最大値: {np.max(np.abs(T3_slope)):.4f} Pa")
print(f"T5 最大値: {np.max(np.abs(T5_slope)):.4f} Pa")
print()
print(f"T1 寄与 (sin²θ × max|T1|):      {sin_t**2      * np.max(np.abs(T1_slope)):.4f} Pa")
print(f"T5 寄与 (2sinθcosθ × max|T5|):  {2*sin_t*cos_t * np.max(np.abs(T5_slope)):.4f} Pa")
print(f"T3 寄与 (cos²θ × max|T3|):      {cos_t**2      * np.max(np.abs(T3_slope)):.4f} Pa")

probe_label = {
    "edge":   "(きずのピッチの端に探触子を配置)",
    "center": "(きずのピッチの中心に探触子を配置)",
}[probe_mode]

# ====== 描画（T1・T3・T5・σ を4段） ======
def make_imshow(ax, data, title, interpolation='nearest'):
    vmax = np.percentile(np.abs(data), 98)
    im = ax.imshow(
        data,
        aspect='auto',
        cmap='bwr',
        vmin=-vmax, vmax=vmax,
        extent=[t_axis[0], t_axis[-1], slope_z_mm[-1], slope_z_mm[0]],
        interpolation=interpolation
    )
    ax.set_ylabel('z 方向 [mm]')
    ax.set_title(title)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.colorbar(im, ax=ax, label='[Pa]')

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle(
    f'くさびの斜面（凸角）での応力成分の確認 {probe_label}\n'
    f'(pitch={f_pitch*1e3:.2f} mm, depth={f_depth*1e3:.2f} mm,  '
    f'θ={np.degrees(np.arctan(mn_d/mn_p)):.1f}°)',
    fontsize=12, y=1.01
)

make_imshow(axes[0], T1_slope,    r'$T_1$（縦方向応力、斜面凸角点）')
make_imshow(axes[1], T3_slope,    r'$T_3$（横方向応力、斜面凸角点）')
make_imshow(axes[2], T5_slope,    r'$T_5$（せん断応力、斜面凸角点）')
make_imshow(axes[3], sigma_slope, r'$\sigma_{tt}=T_1\sin^2\theta+2T_5\sin\theta\cos\theta+T_3\cos^2\theta$')

axes[-1].set_xlabel(r'Time [$\mu\mathrm{s}$]')
plt.tight_layout()

fig_name = os.path.join(
    output_dir,
    f"kusabi_T135_slope_check_{probe_mode}_pitch{int(f_pitch*1e5)}_depth{int(f_depth*1e5)}.png"
)
plt.savefig(fig_name, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_name}")
plt.show()
