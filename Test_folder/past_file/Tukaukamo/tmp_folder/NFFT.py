import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.spatial.distance import cosine

# ============================================================
#  基本パラメータ
# ============================================================

fft_N = 2 ** 14

x_length = 0.02       # [m]
y_length = 0.04       # [m]
mesh_length = 1.0e-5  # [m]

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P波速度
ct = np.sqrt(G / rho)                                    # S波速度

dx = x_length / int(x_length / mesh_length)

# FDTD dt
dt_sim = dx / cl / np.sqrt(6)
print(f"FDTD dt_sim = {dt_sim:.3e} [s]")

# ゲート幅（ピーク周り切り出し）
left  = 2 ** 10
right = (2 ** 10) * 3

exp_gate_start = 0
sim_gate_start = 7560

# ============================================================
#  ユーティリティ
# ============================================================

def kiritori2_safe(data_sample, left, right):
    """
    波形の最大値まわりに [left, right] だけ切り出し（範囲外は安全にクリップ）。
    """
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend], datamaxhere, datamaxstart, datamaxend

def make_fftdata(data, dt_sample):
    """
    data を fft_N 点にゼロパディングして rFFT を行う
    """
    n = fft_N
    if len(data) > n:
        raise ValueError(f"データ長 {len(data)} が fft_N={n} を超えています。")
    howmanyzero = n - len(data)
    left_pad = howmanyzero // 2
    right_pad = howmanyzero - left_pad
    data_fft = np.pad(data, (left_pad, right_pad), mode="constant")
    X = rfft(data_fft)
    freqs = rfftfreq(n, d=dt_sample)
    magnitude = np.abs(X) / (n / 2)
    return magnitude, freqs

def interpolate_to_target_dt(data_raw, dt_original, dt_target):
    """波形 data_raw (dt_original 刻み) を dt_target 刻みに再補間する（1D）。"""
    t_original = np.arange(0, len(data_raw) * dt_original, dt_original)
    t_max = (len(data_raw) - 1) * dt_original
    t_new = np.arange(0, t_max, dt_target)
    func = interp1d(t_original, data_raw, kind='cubic')
    return func(t_new)

def load_experiment_from_folder(target_dir):
    """指定フォルダ内の 'scope_*.csv' をすべて読み込んで平均化する"""
    waves = []
    time_exp = None
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"実験データフォルダが見つかりません: {target_dir}")

    search_path = os.path.join(target_dir, "scope_*.csv")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        raise FileNotFoundError(f"フォルダ内に scope_*.csv が見つかりません: {target_dir}")

    print(f"  Reading {len(file_list)} files from: {os.path.basename(target_dir)}")

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
            
        except Exception as e:
            print(f"    Warning: Error reading {os.path.basename(path)}: {e}")
            continue

    if not waves:
        raise RuntimeError(f"有効な波形データが読み込めませんでした: {target_dir}")

    waves = np.vstack(waves)
    mean_wave = waves.mean(axis=0)
    return time_exp, mean_wave, waves

def smooth_data(data, window_len=10):
    """移動平均"""
    if window_len < 3: return data
    s = np.r_[data[window_len-1:0:-1], data, data[-1:-window_len:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len//2):-(window_len//2)]

def print_metrics(name, data_exp, data_sim):
    """MAEとRMSE, Cosine Similarityを出力"""
    if len(data_exp) != len(data_sim):
        length = min(len(data_exp), len(data_sim))
        data_exp = data_exp[:length]
        data_sim = data_sim[:length]
    
    mae = np.mean(np.abs(data_exp - data_sim))
    rmse = np.sqrt(np.mean((data_exp - data_sim)**2))
    cos_sim = 1.0 - cosine(data_exp, data_sim)
    
    print(f"{name:<12} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | CosSim: {cos_sim:.4f}")

# --- パス生成ヘルパー ---

def get_exp_folder_path(base_root, shape, pitch, depth):
    if shape.lower() == "smooth":
        return os.path.join(base_root, "Smooth", "0_0")

    shape_map = {
        "sankaku": "Sankaku",
        "kusabi":  "Kusabi",
        "hanen":   "Hanen"
    }
    dir_shape = shape_map.get(shape, shape.capitalize())
    p_val = pitch.replace("p", "").replace(".", "")
    try:
        d_num = float(depth.replace("d", ""))
        val_100 = d_num * 100
        if val_100.is_integer():
            d_str = str(int(val_100))
        else:
            d_str = str(val_100)
    except ValueError:
        d_str = depth.replace("d", "").replace(".", "")

    dir_name = f"{p_val}_{d_str}"
    return os.path.join(base_root, dir_shape, dir_name)

def get_sim_folder_path(base_root, shape):
    shape_map = {
        "sankaku": "Sankaku",
        "kusabi":  "Kusabi",
        "hanen":   "Hanen",
        "smooth":  "Smooth"
    }
    dir_shape = shape_map.get(shape, shape.capitalize())
    return os.path.join(base_root, dir_shape)

# ============================================================
#  メイン処理
# ============================================================

if __name__ == "__main__":

    # --- ディレクトリ設定 ---
    doc_path = r"C:\Users\cs16\Documents\Test_folder"
    exp_base_dir = os.path.join(doc_path, "Experiment_Data")
    sim_base_dir = os.path.join(doc_path, "Simulation_Data")

    # ★ 1. 解析・正規化の周波数範囲設定
    freq_min = 2.0e6  # [Hz]
    freq_max = 8.0e6  # [Hz]

    # ★ 2. 実験データの条件選択
    target_shape = "kusabi"
    target_pitch = "p1.50"
    target_depth = "d0.15"
    
    # ★ 3. シミュレーションファイル名
    target_sim_filename = "kusabi_cupy_pitch150_depth15.csv"
    
    # ============================================================

    print("-" * 60)
    print("処理を開始します...")

    # --- パス設定 ---
    target_exp_dir = get_exp_folder_path(exp_base_dir, target_shape, target_pitch, target_depth)
    sim_shape_dir = get_sim_folder_path(sim_base_dir, target_shape)
    target_sim_path = os.path.join(sim_shape_dir, target_sim_filename)

    print(f"【Target】 {target_shape} / {target_pitch} / {target_depth}")
    print(f"  Exp Dir : {target_exp_dir}")
    print(f"  Sim File: {target_sim_path}")

    # --- 読み込み ---
    # Experiment
    time_exp, wave_exp_mean, _ = load_experiment_from_folder(target_exp_dir)

    # Simulation
    if not os.path.exists(target_sim_path):
        raise FileNotFoundError(f"Simulation file not found: {target_sim_path}")
    wave_sim = np.loadtxt(target_sim_path, delimiter=",", dtype=float)

    dt_exp = time_exp[1] - time_exp[0]
    print(f"Estimated dt_exp = {dt_exp:.3e} [s]")

    # リサンプリング (Sim -> Exp の時間刻みに合わせる)
    wave_sim_resampled = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)

    # ゲート処理 (ピーク周辺切り出し)
    # Experiment
    tail_exp = wave_exp_mean[exp_gate_start:]
    seg_exp, _, _, _ = kiritori2_safe(tail_exp, left, right)
    
    # Simulation
    tail_sim = wave_sim_resampled[sim_gate_start:]
    seg_sim, _, _, _ = kiritori2_safe(tail_sim, left, right)

    # FFT計算
    fft_exp, freq = make_fftdata(seg_exp, dt_exp)
    fft_sim, _    = make_fftdata(seg_sim, dt_exp)

    # ----------------------------------------------------------------
    # 特定範囲 (freq_min ~ freq_max) の抽出・正規化
    # ----------------------------------------------------------------
    band = (freq >= freq_min) & (freq <= freq_max)
    f_axis = freq[band] / 1e6  # MHz
    
    # 指定範囲内の生データ振幅
    f_exp_tgt = fft_exp[band]
    f_sim_tgt = fft_sim[band]
    
    # 正規化 (それぞれの波形の指定範囲内の最大値を 1.0 とする)
    norm_exp = f_exp_tgt / np.max(f_exp_tgt)
    norm_sim = f_sim_tgt / np.max(f_sim_tgt)

    # ============================================================
    # ★ グラフ作成: Normalized Amplitude spectrum (Single)
    # ============================================================
    plt.figure(figsize=(8, 5))
    plt.plot(f_axis, norm_exp, label="Exp (Norm)", color='tab:orange')
    plt.plot(f_axis, norm_sim, label="Sim (Norm)", color='tab:blue')
    
    plt.title(f"Normalized Amplitude spectrum ({freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Normalized Amplitude spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ============================================================
    #  評価指標出力
    # ============================================================
    print("\n" + "="*60)
    print(f"一致度評価 (範囲: {freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz)")
    print("-" * 60)
    
    w = 10
    print(f"--- 平滑化 (移動平均: window={w}) ---")
    print_metrics("Spec Shape", smooth_data(norm_exp, w), smooth_data(norm_sim, w))
    
    print("-" * 60)
    print("--- 生データ (平滑化なし) ---")
    print_metrics("Spec Shape", norm_exp, norm_sim)
    
    print("="*60 + "\n")