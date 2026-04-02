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
# DOC_PATH = r"C:\Users\cs16\Documents\Test_folder"
DOC_PATH = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder"   # 自宅PC

EXP_BASE_DIR = os.path.join(DOC_PATH, "Experiment_Data")
SIM_BASE_DIR = os.path.join(DOC_PATH, "Simulation_Data")

# 解析対象
TARGET_SHAPES = ["Hanen"]
TARGET_PITCH_NAME = "125"
TARGET_DEPTH_NAME = "20"
STEP_LIST = [1, 2, 5, 10]

# 周波数範囲
FREQ_MIN = 2.0e6
FREQ_MAX = 8.0e6

# FFTパラメータ
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
        f"smooth_cupy_pitch125_depth0.csv",
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
    if os.path.exists(path_step): return path_step
    
    if step == 1:
        fname_normal = f"{fname_shape}_cupy_pitch{p_val}_depth{d_val}.csv"
        path_normal = os.path.join(sim_base_dir, shape_dir, fname_normal)
        if os.path.exists(path_normal): return path_normal
    return None

def calculate_metrics(data_exp, data_sim):
    """MAE, RMSE のみを計算 (相関係数削除)"""
    # サイズ合わせ
    length = min(len(data_exp), len(data_sim))
    d_e = data_exp[:length]
    d_s = data_sim[:length]
    
    mae = np.mean(np.abs(d_e - d_s))
    rmse = np.sqrt(np.mean((d_e - d_s)**2))
    # corr = np.corrcoef(d_e, d_s)[0, 1]  # 削除
    return mae, rmse

# ============================================================
#  形状ごとの処理関数
# ============================================================
def process_shape_metrics(shape_name):
    print(f"\n[{shape_name}] の評価指標計算を開始します...")

    # --- 1. 実験データ ---
    exp_defect_dir = os.path.join(EXP_BASE_DIR, shape_name, f"{TARGET_PITCH_NAME}_{TARGET_DEPTH_NAME}")
    exp_smooth_dir = os.path.join(EXP_BASE_DIR, "Smooth", "0_0")
    
    time_exp, wave_exp_defect = load_experiment_from_folder(exp_defect_dir)
    _, wave_exp_smooth = load_experiment_from_folder(exp_smooth_dir)
    
    if time_exp is None or wave_exp_smooth is None:
        print(f"  [Skip] Experiment data missing for {shape_name}")
        return None

    dt_exp = time_exp[1] - time_exp[0]
    seg_exp_def = kiritori2_safe(wave_exp_defect[EXP_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    seg_exp_sm  = kiritori2_safe(wave_exp_smooth[EXP_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    
    fft_exp_def, freq_axis_exp = make_fftdata(seg_exp_def, dt_exp)
    fft_exp_sm, _ = make_fftdata(seg_exp_sm, dt_exp)

    band = (freq_axis_exp >= FREQ_MIN) & (freq_axis_exp <= FREQ_MAX)
    
    # 正規化 (Rawのみ)
    exp_ratio_raw = fft_exp_def[band] / (fft_exp_sm[band] + 1e-12)
    exp_norm_raw  = exp_ratio_raw / np.max(exp_ratio_raw)
    
    # --- 2. Sim Smooth (基準波形) ---
    smooth_sim_path = get_smooth_sim_path(SIM_BASE_DIR, TARGET_PITCH_NAME)
    if not smooth_sim_path:
        print("  [Error] Simulation Smooth file missing.")
        return None
    wave_sim_smooth = np.loadtxt(smooth_sim_path, delimiter=",", dtype=float)
    wave_sim_smooth_res = interpolate_to_target_dt(wave_sim_smooth, dt_sim, dt_exp)
    seg_sim_sm = kiritori2_safe(wave_sim_smooth_res[SIM_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    fft_sim_sm, _ = make_fftdata(seg_sim_sm, dt_exp)
    
    # --- 3. Step Loop ---
    results_raw = []   # (step, mae, rmse)

    for step in STEP_LIST:
        fpath = get_step_sim_filepath(SIM_BASE_DIR, shape_name, TARGET_PITCH_NAME, TARGET_DEPTH_NAME, step)
        if not fpath:
            continue
            
        try:
            wave_sim = np.loadtxt(fpath, delimiter=",", dtype=float)
            wave_sim_res = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)
            seg_sim = kiritori2_safe(wave_sim_res[SIM_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
            fft_sim, _ = make_fftdata(seg_sim, dt_exp)
            
            # 正規化
            sim_ratio_raw = fft_sim[band] / (fft_sim_sm[band] + 1e-12)
            sim_norm_raw  = sim_ratio_raw / np.max(sim_ratio_raw)
            
            # 指標計算 (相関係数削除)
            mae_r, rmse_r = calculate_metrics(exp_norm_raw, sim_norm_raw)
            results_raw.append((step, mae_r, rmse_r))
            
        except Exception as e:
            print(f"  [Error] Step {step}: {e}")

    if not results_raw:
        return None

    # --- 4. Plot (Individual Shape) ---
    steps = [x[0] for x in results_raw]
    raw_mae  = [x[1] for x in results_raw]
    raw_rmse = [x[2] for x in results_raw]
    
    # 2つのグラフ (MAE, RMSE) に変更
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_ticks = steps
    
    # 1. MAE
    axes[0].plot(steps, raw_mae, 'o-', color='tab:blue')
    axes[0].set_title("MAE")
    axes[0].set_xlabel("Step Size")
    axes[0].set_ylabel("MAE")
    axes[0].set_xticks(x_ticks)
    axes[0].grid(True)

    # 2. RMSE
    axes[1].plot(steps, raw_rmse, 's-', color='tab:orange')
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Step Size")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(x_ticks)
    axes[1].grid(True)

    fig.suptitle(f"Evaluation Step Size (Defect/Smooth) : {shape_name} P{TARGET_PITCH_NAME} D{TARGET_DEPTH_NAME}", fontsize=16)
    plt.tight_layout()
    
    save_name = f"5_Step_Comparison_{shape_name}.png"
    plt.savefig(save_name, dpi=150)
    print(f"  -> Saved: {save_name}")
    plt.close(fig)

    # 後でまとめるためにデータを返す
    return steps, raw_mae, raw_rmse

def plot_combined_metrics(all_data):
    """
    全形状のデータをまとめてプロットする関数
    all_data: {shape_name: (steps, mae, rmse), ...}
    """
    print(f"\n[Combined] 全形状の比較グラフを作成します...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # マーカーや色を変えるためのリスト
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    shapes = list(all_data.keys())
    
    for i, shape in enumerate(shapes):
        steps, mae, rmse = all_data[shape]
        
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]
        
        # MAE
        axes[0].plot(steps, mae, marker=m, color=c, label=shape, linestyle='-')
        # RMSE
        axes[1].plot(steps, rmse, marker=m, color=c, label=shape, linestyle='-')

    # MAE Plot Settings
    axes[0].set_title("MAE")
    axes[0].set_xlabel("Step Size")
    axes[0].set_ylabel("MAE")
    axes[0].set_xticks(STEP_LIST)
    axes[0].grid(True)
    axes[0].legend()

    # RMSE Plot Settings
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Step Size")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(STEP_LIST)
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"Step Size Evaluation (Defect/Smooth) - All Shapes Comparison (P{TARGET_PITCH_NAME} D{TARGET_DEPTH_NAME})", fontsize=16)
    plt.tight_layout()
    
    save_name = "5_Step_Comparison_All_Shapes.png"
    plt.savefig(save_name, dpi=150)
    print(f"  -> Saved: {save_name}")
    plt.close(fig)

# ============================================================
#  Main
# ============================================================
def main():
    print("評価指標の推移グラフ作成を開始します...")
    
    all_data = {} # 全形状のデータを保存する辞書

    for shape in TARGET_SHAPES:
        result = process_shape_metrics(shape)
        if result is not None:
            # 戻り値を保存 (steps, mae, rmse)
            all_data[shape] = result
            
    # 全形状のデータが集まったら比較グラフを作成
    if all_data:
        plot_combined_metrics(all_data)
    else:
        print("プロットするデータがありませんでした。")

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()