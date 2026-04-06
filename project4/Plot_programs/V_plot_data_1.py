import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math
from scipy.interpolate import interp1d
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.spatial.distance import cosine

# ============================================================
#  設定 (環境に合わせて変更してください)
# ============================================================
DOC_PATH = r"C:\Users\cs16\Roughness\project4"
# DOC_PATH= r"C:/Users/hisay/OneDrive/ドキュメント/test_folder"   # 自宅PC

FREQ_MIN = 2.0e6  # [Hz]
FREQ_MAX = 8.0e6  # [Hz]

FFT_N = 2 ** 14
X_LENGTH = 0.02
MESH_LENGTH = 1.0e-5
RHO = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

LEFT_OFFSET  = 2 ** 10
RIGHT_OFFSET = (2 ** 10) * 3
EXP_GATE_START = 0
SIM_GATE_START = 7560

# ============================================================
#  計算用定数・関数定義
# ============================================================
cl = np.sqrt(E / RHO * (1 - V) / (1 + V) / (1 - 2 * V))
dt_sim = (X_LENGTH / int(X_LENGTH / MESH_LENGTH)) / cl / np.sqrt(6)

def kiritori2_safe(data_sample, left, right):
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend]

def make_fftdata(data, dt_sample):
    n = FFT_N
    if len(data) > n:
        data = data[:n]
    
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
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Folder not found: {target_dir}")

    search_path = os.path.join(target_dir, "scope_*.csv")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        raise FileNotFoundError(f"No csv files in: {target_dir}")

    for path in file_list:
        try:
            data = np.genfromtxt(path, delimiter=",", skip_header=2)
            if data.ndim == 1: continue
            data = data[~np.isnan(data).any(axis=1)]
            t = data[:, 0]
            v = data[:, 1]
            if time_exp is None:
                time_exp = t
            
            mode_val = pd.Series(v).mode().iloc[0]
            v_detrended = v - mode_val
            waves.append(v_detrended)
        except:
            continue

    if not waves:
        raise RuntimeError("No valid waves found.")

    min_len = min(len(w) for w in waves)
    if len(time_exp) > min_len:
        time_exp = time_exp[:min_len]
    waves_trimmed = [w[:min_len] for w in waves]
    waves = np.vstack(waves_trimmed)
    mean_wave = waves.mean(axis=0)
    return time_exp, mean_wave

def get_sim_filepath(sim_base_dir, shape, p_val, d_str):
    shape_map = {
        "Sankaku": "sankaku", "sankaku": "sankaku",
        "Kusabi": "kusabi", "kusabi": "kusabi",
        "Hanen": "hanen", "hanen": "hanen" , 
        "Kukei": "kukei", "kukei": "kukei"
    }
    s_name = shape_map.get(shape, shape.lower())
    dir_shape = shape.capitalize() if shape.lower() != "smooth" else "Smooth"
    filename = f"{s_name}_cupy_pitch{p_val}_depth{d_str}.csv"
    return os.path.join(sim_base_dir, dir_shape, filename)

def get_smooth_sim_path(sim_base_dir, p_val):
    candidates = [
        f"smooth_cupy_pitch125_depth0.csv",
    ]
    for fname in candidates:
        path = os.path.join(sim_base_dir, "Smooth", fname)
        if os.path.exists(path):
            return path
    return None

def process_one_case(exp_dir, shape, p_val, d_val, exp_base_dir, sim_base_dir, wave_exp_smooth_mean):
    sim_path = get_sim_filepath(sim_base_dir, shape, p_val, d_val)
    smooth_sim_path = get_smooth_sim_path(sim_base_dir, p_val)
    
    if not os.path.exists(sim_path):
        return None

    try:
        time_exp, wave_exp_mean = load_experiment_from_folder(exp_dir)
        wave_sim = np.loadtxt(sim_path, delimiter=",", dtype=float)
        
        if smooth_sim_path and os.path.exists(smooth_sim_path):
            wave_sim_smooth = np.loadtxt(smooth_sim_path, delimiter=",", dtype=float)
        else:
            wave_sim_smooth = np.ones_like(wave_sim)
    except Exception:
        return None

    dt_exp = time_exp[1] - time_exp[0]
    
    wave_sim_res = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)
    wave_sim_smooth_res = interpolate_to_target_dt(wave_sim_smooth, dt_sim, dt_exp)
    
    seg_exp = kiritori2_safe(wave_exp_mean[EXP_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    seg_sim = kiritori2_safe(wave_sim_res[SIM_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    seg_exp_smooth = kiritori2_safe(wave_exp_smooth_mean[EXP_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)
    seg_sim_smooth = kiritori2_safe(wave_sim_smooth_res[SIM_GATE_START:], LEFT_OFFSET, RIGHT_OFFSET)

    fft_exp, freq = make_fftdata(seg_exp, dt_exp)
    fft_sim, _    = make_fftdata(seg_sim, dt_exp)
    fft_exp_sm, _ = make_fftdata(seg_exp_smooth, dt_exp)
    fft_sim_sm, _ = make_fftdata(seg_sim_smooth, dt_exp)
    
    band = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
    f_axis = freq[band] / 1e6
    
    ratio_exp = fft_exp[band] / (fft_exp_sm[band] + 1e-12)
    ratio_sim = fft_sim[band] / (fft_sim_sm[band] + 1e-12)
    
    norm_ratio_exp = ratio_exp / np.max(ratio_exp)
    norm_ratio_sim = ratio_sim / np.max(ratio_sim)
    
    return {
        "freq": f_axis,
        "exp": norm_ratio_exp,
        "sim": norm_ratio_sim,
        "title": f"Pitch:{p_val} Depth:{d_val}",
        "shape": shape
    }

def main():
    exp_base_dir = os.path.join(DOC_PATH, "Experiment_Data")
    sim_base_dir = os.path.join(DOC_PATH, "Simulation_Data")
    
    print("処理を開始します...")

    smooth_exp_dir = os.path.join(exp_base_dir, "Smooth", "0_0")
    if not os.path.exists(smooth_exp_dir):
        print(f"Error: Smooth folder not found at {smooth_exp_dir}")
        return
        
    try:
        _, wave_exp_smooth_mean = load_experiment_from_folder(smooth_exp_dir)
    except Exception as e:
        print(f"Smoothデータの読み込みに失敗しました: {e}")
        return

    results_by_shape = {}

    if os.path.exists(exp_base_dir):
        shape_dirs = [d for d in os.listdir(exp_base_dir) if os.path.isdir(os.path.join(exp_base_dir, d))]
        
        for shape in shape_dirs:
            if shape.lower() == "smooth": continue 
            
            print(f"Searching shape: {shape} ...")
            shape_path = os.path.join(exp_base_dir, shape)
            case_dirs = [d for d in os.listdir(shape_path) if os.path.isdir(os.path.join(shape_path, d))]
            
            for case in case_dirs:
                if "_" not in case: continue
                parts = case.split("_")
                if len(parts) != 2: continue
                p_val, d_val = parts[0], parts[1]
                
                res = process_one_case(
                    os.path.join(shape_path, case), shape, p_val, d_val,
                    exp_base_dir, sim_base_dir, wave_exp_smooth_mean
                )
                
                if res:
                    if shape not in results_by_shape:
                        results_by_shape[shape] = []
                    results_by_shape[shape].append(res)
    else:
        print("実験データフォルダが見つかりません。")
        return

    if not results_by_shape:
        print("表示できる結果がありませんでした。")
        return

    # --- グラフ描画ループ ---
    for shape_name, case_list in results_by_shape.items():
        print(f"Plotting results for: {shape_name} ({len(case_list)} cases)")
        
        # ソート
        case_list.sort(key=lambda x: x['title'])

        num_plots = len(case_list)
        cols = 3
        rows = math.ceil(num_plots / cols)
        
        # 図の作成
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
        
        for i, ax in enumerate(axes_flat):
            if i < num_plots:
                d = case_list[i]
                
                # 指定の凡例と色でプロット
                ax.plot(d['freq'], d['exp'], label="Exp (Max=1)", color='tab:orange')
                ax.plot(d['freq'], d['sim'], label="Sim (Max=1)", color='tab:blue')
                
                # サブプロットのタイトル (条件名)
                ax.set_title(d['title'], fontsize=11)
                
                # グリッド
                ax.grid(True)
                ax.set_ylim(0, 1.1)
                
                # 軸ラベル (ご指定の内容)
                # 混み合うのを避けるため、Y軸ラベルは左端、X軸ラベルは下段のみに設定するのが一般的ですが
                # ご指定通りにするため、各グラフが十分離れるように figsize を少し大きくしています。
                if i % cols == 0:
                    ax.set_ylabel("Normalized Amplitude spectrum (Defect/Smooth)")
                
                if i >= num_plots - cols:
                    ax.set_xlabel("Frequency [MHz]")

                # 最初のグラフにのみ凡例を表示 (全部につけると見づらいため。必要ならifを外してください)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=9)

            else:
                ax.axis('off')

        # 全体のタイトル (ご指定のタイトル形式 + 形状名)
        main_title = f"[{shape_name}] Normalized Amplitude spectrum (Defect/Smooth) ({FREQ_MIN/1e6:.1f}-{FREQ_MAX/1e6:.1f} MHz)"
        fig.suptitle(main_title, fontsize=16)
        
        save_name = f"1_Comparison_{shape_name}.png"
        plt.savefig(save_name, dpi=150)
        print(f"  -> Saved: {save_name}")
        plt.close(fig)

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()