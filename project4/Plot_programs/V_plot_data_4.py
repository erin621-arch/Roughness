import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

# ============================================================
#  設定
# ============================================================
DOC_PATH = r"C:\Users\cs16\Roughness\project4"
# DOC_PATH = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder"   # 自宅PC

EXP_BASE_DIR = os.path.join(DOC_PATH, "Experiment_Data")
SIM_BASE_DIR = os.path.join(DOC_PATH, "Simulation_Data")

# 解析対象の形状リスト
TARGET_SHAPES = ["Sankaku", "Kusabi", "Hanen"]

# 条件設定 (Pitch 1.25, Depth 0.2)
TARGET_PITCH_NAME = "125"
TARGET_DEPTH_NAME = "20"

# 比較するステップサイズ
STEP_LIST = [1, 2, 5, 10]

# 解析周波数範囲
FREQ_MIN = 2.0e6
FREQ_MAX = 8.0e6

# パラメータ
FFT_N = 2 ** 14
X_LENGTH = 0.02
MESH_LENGTH = 1.0e-5
RHO = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

# ゲートパラメータ
LEFT_OFFSET  = 2 ** 10
RIGHT_OFFSET = (2 ** 10) * 3
EXP_GATE_START = 0
SIM_GATE_START = 7560

# ============================================================
#  計算関数
# ============================================================
cl = np.sqrt(E / RHO * (1 - V) / (1 + V) / (1 - 2 * V))
dx = X_LENGTH / int(X_LENGTH / MESH_LENGTH)
dt_sim = dx / cl / np.sqrt(6)

def kiritori2_safe(data_sample, left, right):
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend]

def make_fftdata(data, dt_sample):
    n = FFT_N
    if len(data) > n: data = data[:n]
    howmanyzero = n - len(data)
    left_pad = howmanyzero // 2
    right_pad = howmanyzero - left_pad
    data_fft = np.pad(data, (left_pad, right_pad), mode="constant")
    X = rfft(data_fft)
    freqs = rfftfreq(n, d=dt_sample)
    magnitude = np.abs(X) / (n / 2)
    return magnitude, freqs

def interpolate_to_target_dt(data_raw, dt_original, dt_target):
    t_original = np.arange(0, len(data_raw) * dt_original, dt_original)
    t_max = (len(data_raw) - 1) * dt_original
    t_new = np.arange(0, t_max, dt_target)
    func = interp1d(t_original, data_raw, kind='cubic')
    return func(t_new)

def load_experiment_from_folder(target_dir):
    waves = []
    time_exp = None
    if not os.path.exists(target_dir): return None, None
    search_path = os.path.join(target_dir, "scope_*.csv")
    file_list = sorted(glob.glob(search_path))
    if not file_list: return None, None

    for path in file_list:
        try:
            data = np.genfromtxt(path, delimiter=",", skip_header=2)
            if data.ndim == 1: continue
            data = data[~np.isnan(data).any(axis=1)]
            t = data[:, 0]
            v = data[:, 1]
            if time_exp is None: time_exp = t
            mode_val = pd.Series(v).mode().iloc[0]
            waves.append(v - mode_val)
        except: continue

    if not waves: return None, None
    min_len = min(len(w) for w in waves)
    if len(time_exp) > min_len: time_exp = time_exp[:min_len]
    waves = np.vstack([w[:min_len] for w in waves])
    return time_exp, waves.mean(axis=0)

def get_smooth_sim_path(sim_base_dir, p_val_str):
    candidates = [
        f"smooth_cupy_pitch125_depth0.csv"
    ]
    for fname in candidates:
        path = os.path.join(sim_base_dir, "Smooth", fname)
        if os.path.exists(path): return path
    return None

def get_step_sim_filepath(sim_base_dir, shape, p_val, d_val, step):
    shape_dir = shape.capitalize()
    fname_shape = shape.lower()
    
    fname_step = f"{fname_shape}_cupy_pitch{p_val}_depth{d_val}_step{step}.csv"
    path_step = os.path.join(sim_base_dir, shape_dir, fname_step)
    
    if os.path.exists(path_step):
        return path_step, f"Step {step}"
    
    if step == 1:
        fname_normal = f"{fname_shape}_cupy_pitch{p_val}_depth{d_val}.csv"
        path_normal = os.path.join(sim_base_dir, shape_dir, fname_normal)
        if os.path.exists(path_normal):
            return path_normal, f"Step {step} (No Suffix)"

    return None, None

# ============================================================
#  形状ごとの処理関数
# ============================================================
def process_shape(shape_name):
    print(f"\n[{shape_name}] の処理を開始します...")

    # --- 1. 実験データの読み込み ---
    exp_defect_dir = os.path.join(EXP_BASE_DIR, shape_name, f"{TARGET_PITCH_NAME}_{TARGET_DEPTH_NAME}")
    exp_smooth_dir = os.path.join(EXP_BASE_DIR, "Smooth", "0_0")
    
    time_exp, wave_exp_defect = load_experiment_from_folder(exp_defect_dir)
    if time_exp is None:
        print(f"  [Skip] Experiment Defect data not found: {exp_defect_dir}")
        return

    _, wave_exp_smooth = load_experiment_from_folder(exp_smooth_dir)
    if wave_exp_smooth is None:
        print("  [Error] Experiment Smooth data not found.")
        return

    dt_exp = time_exp[1] - time_exp[0]

    # 実験データ FFT
    seg_exp_def = kiritori2_safe(wave_exp_defect[EXP_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    seg_exp_sm  = kiritori2_safe(wave_exp_smooth[EXP_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    
    fft_exp_def, freq_axis_exp = make_fftdata(seg_exp_def, dt_exp)
    fft_exp_sm, _ = make_fftdata(seg_exp_sm, dt_exp)

    # バンド抽出
    band = (freq_axis_exp >= FREQ_MIN) & (freq_axis_exp <= FREQ_MAX)
    f_band = freq_axis_exp[band] / 1e6
    
    # 実験データ正規化
    exp_spec_tgt = fft_exp_def[band]
    exp_norm_spec = exp_spec_tgt / np.max(exp_spec_tgt)
    
    exp_ratio_tgt = fft_exp_def[band] / (fft_exp_sm[band] + 1e-12)
    exp_norm_ratio = exp_ratio_tgt / np.max(exp_ratio_tgt)


    # --- 2. シミュレーションデータの読み込み ---
    smooth_sim_path = get_smooth_sim_path(SIM_BASE_DIR, TARGET_PITCH_NAME)
    if not smooth_sim_path:
        print("  [Error] Simulation Smooth file not found.")
        return
    
    wave_sim_smooth = np.loadtxt(smooth_sim_path, delimiter=",", dtype=float)
    wave_sim_smooth_res = interpolate_to_target_dt(wave_sim_smooth, dt_sim, dt_exp)
    seg_sim_sm = kiritori2_safe(wave_sim_smooth_res[SIM_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    fft_sim_sm, _ = make_fftdata(seg_sim_sm, dt_exp)

    sim_results = {}
    for step in STEP_LIST:
        fpath, label = get_step_sim_filepath(SIM_BASE_DIR, shape_name, TARGET_PITCH_NAME, TARGET_DEPTH_NAME, step)
        
        if not fpath:
            print(f"  [Warning] Step {step} file not found for {shape_name}")
            continue
            
        try:
            wave_sim = np.loadtxt(fpath, delimiter=",", dtype=float)
            wave_sim_res = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)
            seg_sim = kiritori2_safe(wave_sim_res[SIM_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
            fft_sim, _ = make_fftdata(seg_sim, dt_exp)
            
            sim_spec_tgt = fft_sim[band]
            sim_norm_spec = sim_spec_tgt / np.max(sim_spec_tgt)
            
            sim_ratio_tgt = fft_sim[band] / (fft_sim_sm[band] + 1e-12)
            sim_norm_ratio = sim_ratio_tgt / np.max(sim_ratio_tgt)
            
            sim_results[step] = {"spec": sim_norm_spec, "ratio": sim_norm_ratio}
            print(f"  Loaded: {label}")
            
        except Exception as e:
            print(f"  [Error] Failed to process Step {step}: {e}")

    if not sim_results:
        print("  有効なシミュレーションデータがありません。")
        return

    # ============================================================
    #  グラフ描画 (左右に配置)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    colors = {1: 'tab:blue', 5: 'tab:green', 10: 'tab:purple'}
    linestyles = {1: '-', 5: '--', 10: '-.'}

    # --- 左: 単体スペクトル ---
    ax1 = axes[0]
    ax1.plot(f_band, exp_norm_spec, label="Experiment", color='red', linewidth=2.5)
    for step in sorted(sim_results.keys()):
        d = sim_results[step]
        ax1.plot(f_band, d['spec'], 
                 label=f"Sim Step {step}", 
                 color=colors.get(step, 'black'),
                 linestyle=linestyles.get(step, '-'),
                 linewidth=1.5)
    
    ax1.set_title(f"Normalized Amplitude spectrum (Effect of Step Size)\n({shape_name} P{TARGET_PITCH_NAME.replace('p','')} D{TARGET_DEPTH_NAME.replace('d','')}, {FREQ_MIN/1e6:.1f}-{FREQ_MAX/1e6:.1f} MHz)")
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel("Normalized Amplitude spectrum")
    ax1.legend()
    ax1.grid(True)

    # --- 右: 振幅比 ---
    ax2 = axes[1]
    ax2.plot(f_band, exp_norm_ratio, label="Experiment", color='red', linewidth=2.5)
    for step in sorted(sim_results.keys()):
        d = sim_results[step]
        ax2.plot(f_band, d['ratio'], 
                 label=f"Sim Step {step}", 
                 color=colors.get(step, 'black'),
                 linestyle=linestyles.get(step, '-'),
                 linewidth=1.5)
    
    ax2.set_title(f"Normalized Amplitude spectrum (Defect/Smooth) (Effect of Step Size)\n({shape_name} P{TARGET_PITCH_NAME.replace('p','')} D{TARGET_DEPTH_NAME.replace('d','')}, {FREQ_MIN/1e6:.1f}-{FREQ_MAX/1e6:.1f} MHz)")
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Normalized Amplitude spectrum (Defect/Smooth)")
    ax2.legend()
    ax2.grid(True)

    # レイアウト調整
    plt.tight_layout()
    
    # ファイル保存
    save_name = f"4_Step_Comparison_{shape_name}.png"
    plt.savefig(save_name, dpi=150)
    print(f"  -> Saved: {save_name}")
    plt.close(fig)

# ============================================================
#  メイン処理
# ============================================================
def main():
    print("一括処理を開始します...")
    for shape in TARGET_SHAPES:
        process_shape(shape)
    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()