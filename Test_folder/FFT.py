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
    戻り値:
        切り出しデータ, ピーク位置index, 切り出し開始index, 切り出し終了index
    """
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend], datamaxhere, datamaxstart, datamaxend

def make_fftdata(data, dt_sample):
    """
    data を fft_N 点にゼロパディングして rFFT を行う（周波数軸は正確に rfftfreq）。
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
    """
    指定フォルダ内の 'scope_*.csv' をすべて読み込んで平均化する
    """
    waves = []
    time_exp = None
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"実験データフォルダが見つかりません: {target_dir}")

    # フォルダ内の scope_*.csv をすべて取得
    search_path = os.path.join(target_dir, "scope_*.csv")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        raise FileNotFoundError(f"フォルダ内に scope_*.csv が見つかりません: {target_dir}")

    print(f"  Reading {len(file_list)} files from: {os.path.basename(target_dir)}")

    for path in file_list:
        try:
            data = np.genfromtxt(path, delimiter=",", skip_header=2)
            if data.ndim == 1: continue

            # NaN除去
            data = data[~np.isnan(data).any(axis=1)]
            t = data[:, 0]
            v = data[:, 1]

            if time_exp is None:
                time_exp = t
            
            # ベースライン補正
            mode_val = pd.Series(v).mode().iloc[0]
            v_detrended = v - mode_val
            waves.append(v_detrended)
            
        except Exception as e:
            print(f"    Warning: Error reading {os.path.basename(path)}: {e}")
            continue

    if not waves:
        raise RuntimeError(f"有効な波形データが読み込めませんでした: {target_dir}")

    # 全波形の最小長さを取得
    min_len = min(len(w) for w in waves)
    
    # 時間軸が最小長さより長い場合があるためカット
    if len(time_exp) > min_len:
        time_exp = time_exp[:min_len]

    # すべての波形を最小長さに合わせてカット（トリミング）
    waves_trimmed = [w[:min_len] for w in waves]

    # vstackを実行
    waves = np.vstack(waves_trimmed)

    mean_wave = waves.mean(axis=0)
    return time_exp, mean_wave, waves

def smooth_data(data, window_len=10):
    """
    移動平均でスペクトルを滑らかにする（ノイズ除去用）
    window_len: 平均するデータ点数（大きいほど滑らかになる）
    """
    if window_len < 3: return data
    s = np.r_[data[window_len-1:0:-1], data, data[-1:-window_len:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len//2):-(window_len//2)]

def print_metrics(name, data_exp, data_sim):
    """
    MAEとRMSEの両方を計算して出力する関数
    """
    if len(data_exp) != len(data_sim):
        length = min(len(data_exp), len(data_sim))
        data_exp = data_exp[:length]
        data_sim = data_sim[:length]
    
    # MAE & RMSE
    mae = np.mean(np.abs(data_exp - data_sim))
    rmse = np.sqrt(np.mean((data_exp - data_sim)**2))
    
    # Cosine Similarity (1.0 = 一致)
    cos_sim = 1.0 - cosine(data_exp, data_sim)
    
    print(f"{name:<12} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | CosSim: {cos_sim:.4f}")

# --- パス生成ヘルパー ---

def get_exp_folder_path(base_root, shape, pitch, depth):
    """実験データのフォルダパスを自動生成"""
    
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
    """シミュレーションデータの格納フォルダ"""
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

    # --- ディレクトリ設定 (お使いの環境に合わせて固定) ---
    # doc_path = r"C:/Users/cs16/Roughness/Test_folder"  # 研究室PC
    doc_path = r"C:/Users/hisay/OneDrive/ドキュメント/Test_folder"   # 自宅PC
    
    exp_base_dir = os.path.join(doc_path, "Experiment_Data")
    sim_base_dir = os.path.join(doc_path, "Simulation_Data")

    # ============================================================
    # ★ 1. 解析・正規化の周波数範囲設定
    # ============================================================
    freq_min = 2.0e6  # [Hz]
    freq_max = 8.0e6  # [Hz]

    # ============================================================
    # ★ 2. 実験データの条件選択
    # ============================================================
    target_shape = "Kusabi"   # "sankaku" / "kusabi" / "hanen"
    target_pitch = "p1.25"     # "p1.25" ...
    target_depth = "d0.20"     # "d0.10" (フォルダ名20), "d0.125" (フォルダ名12.5)
    
    # ============================================================
    # ★ 3. シミュレーションファイル名の指定（手動入力）
    # ============================================================
    
    # target_sim_filename = "sankaku_cupy_pitch200_depth10.csv"
    # target_sim_filename = "sankaku_cupy_pitch125_depth20_step1.csv"

    target_sim_filename = "kusabi2_cupy_pitch125_depth20.csv"
    # target_sim_filename = "sankaku_cupy_pitch125_depth20_step1.csv"

    # target_sim_filename = "hanen_cupy_pitch125_depth20.csv"
    # target_sim_filename = "sankaku_cupy_pitch125_depth20_step1.csv"
    
    # ============================================================

    print("-" * 60)
    print("処理を開始します...")

    # --- パス設定 ---
    target_exp_dir = get_exp_folder_path(exp_base_dir, target_shape, target_pitch, target_depth)
    sim_shape_dir = get_sim_folder_path(sim_base_dir, target_shape)
    target_sim_path = os.path.join(sim_shape_dir, target_sim_filename)
    smooth_exp_dir = get_exp_folder_path(exp_base_dir, "smooth", "", "") 
    p_val = target_pitch.replace("p", "").replace(".", "")
    fname_smooth_sim = f"kukei_ori_cupy_pitch{p_val}_depth0.csv"
    smooth_sim_path = os.path.join(sim_base_dir, "Smooth", fname_smooth_sim)
    
    if not os.path.exists(smooth_sim_path):
        alt_name = f"smooth_cupy_pitch{p_val}_depth0.csv"
        smooth_sim_path = os.path.join(sim_base_dir, "Smooth", alt_name)
    if not os.path.exists(smooth_sim_path):
        fname_backup = "kukei_ori_cupy_pitch125_depth0.csv"
        smooth_sim_path = os.path.join(sim_base_dir, "Smooth", fname_backup)

    print(f"【Target】 {target_shape} / {target_pitch} / {target_depth}")
    print(f"  Exp Dir : {target_exp_dir}")
    print(f"  Sim File: {target_sim_path}")

    # --- 読み込み ---
    time_exp, wave_exp_mean, _ = load_experiment_from_folder(target_exp_dir)
    _, wave_exp_smooth_mean, _ = load_experiment_from_folder(smooth_exp_dir)

    wave_sim = np.loadtxt(target_sim_path, delimiter=",", dtype=float)
    if not os.path.exists(smooth_sim_path):
         print("Warning: Smooth simulation not found!")
         wave_sim_smooth = np.ones_like(wave_sim)
    else:
        wave_sim_smooth = np.loadtxt(smooth_sim_path, delimiter=",", dtype=float)

    dt_exp = time_exp[1] - time_exp[0]
    print(f"Estimated dt_exp = {dt_exp:.3e} [s]")

    wave_sim_resampled = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)
    wave_sim_smooth_resampled = interpolate_to_target_dt(wave_sim_smooth, dt_sim, dt_exp)

    # 時間軸作成 (us単位)
    t_exp_us = np.arange(len(wave_exp_mean)) * dt_exp * 1e6
    t_sim_us = np.arange(len(wave_sim_resampled)) * dt_exp * 1e6

    # ゲート処理
    tail_exp = wave_exp_mean[exp_gate_start:]
    seg_exp, peak_idx_local_exp, seg_start_local_exp, seg_end_local_exp = kiritori2_safe(tail_exp, left, right)
    
    tail_sim = wave_sim_resampled[sim_gate_start:]
    seg_sim, peak_idx_local_sim, seg_start_local_sim, seg_end_local_sim = kiritori2_safe(tail_sim, left, right)

    tail_exp_smooth = wave_exp_smooth_mean[exp_gate_start:]
    seg_exp_smooth, _, _, _ = kiritori2_safe(tail_exp_smooth, left, right)

    tail_sim_smooth = wave_sim_smooth_resampled[sim_gate_start:]
    seg_sim_smooth, _, _, _ = kiritori2_safe(tail_sim_smooth, left, right)

    # グラフ用グローバルインデックス計算
    def get_global_time(local_idx, gate_start, dt):
        return (gate_start + local_idx) * dt * 1e6
    
    

    # ------------------------------------------------------------
    # グラフ表示
    # ------------------------------------------------------------
    
    # 1. 実験波形 (全体)
    plt.figure(figsize=(10, 4))
    plt.plot(t_exp_us, wave_exp_mean, label="experiment defect", color='tab:orange')
    min_len = min(len(t_exp_us), len(wave_exp_smooth_mean))
    # plt.plot(t_exp_us[:min_len], wave_exp_smooth_mean[:min_len], label="experiment smooth", color='gray', alpha=0.7, linestyle="--")
    
    # ゲート線表示
    # plt.axvline(exp_gate_start * dt_exp * 1e6, color="g", linestyle=":", label="Search Start")
    plt.axvline(get_global_time(peak_idx_local_exp, exp_gate_start, dt_exp), color="r", linestyle="--", label="Peak")
    gate_start_t = get_global_time(seg_start_local_exp, exp_gate_start, dt_exp)
    gate_end_t = get_global_time(seg_end_local_exp, exp_gate_start, dt_exp)
    plt.axvspan(gate_start_t, gate_end_t, color="gray", alpha=0.2, label="Gate")
    
    # [単位変更] Voltage [V]
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"[Experiment] Waveform: {target_shape} {target_pitch} {target_depth}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 2. シミュレーション波形 (全体)
    plt.figure(figsize=(10, 4))
    plt.plot(t_sim_us, wave_sim_resampled, label="simulation defect", color='tab:blue')
    # plt.plot(t_sim_us, wave_sim_smooth_resampled, label="simulation smooth", color='gray', alpha=0.7, linestyle="--")
    
    # plt.axvline(sim_gate_start * dt_exp * 1e6, color="g", linestyle=":", label="Search Start")
    plt.axvline(get_global_time(peak_idx_local_sim, sim_gate_start, dt_exp), color="r", linestyle="--", label="Peak")
    gate_start_sim_t = get_global_time(seg_start_local_sim, sim_gate_start, dt_exp)
    gate_end_sim_t = get_global_time(seg_end_local_sim, sim_gate_start, dt_exp)
    plt.axvspan(gate_start_sim_t, gate_end_sim_t, color="gray", alpha=0.2, label="Gate")
    
    # [単位変更] Pressure [Pa]
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Pressure [Pa]")
    plt.title(f"[Simulation] Waveform : {target_shape} {target_pitch} {target_depth}") # Resampled 
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------------------
    # ゲート内波形の比較グラフ
    # --------------------------------------------------------------------------------
    
    # 開始時間を 0 に揃える（相対時間）
    t_rel_exp = np.arange(len(seg_exp)) * dt_exp * 1e6
    t_rel_sim = np.arange(len(seg_sim)) * dt_exp * 1e6

    # 3. ゲート内波形（上下に並べて表示）
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True) 
    
    # 上段: 実験 (オレンジ)
    axes[0].plot(t_rel_exp, seg_exp, color='tab:orange', label="Experiment Defect (Gated)")
    # [単位変更] Voltage [V]
    axes[0].set_ylabel("Voltage [V]")
    # axes[0].set_title("Gated Waveform (Separate)")
    axes[0].set_title("Gated Waveform")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # 下段: シミュレーション (青)
    axes[1].plot(t_rel_sim, seg_sim, color='tab:blue', label="Simulation Defect (Gated)")
    # [単位変更] Pressure [Pa]
    axes[1].set_ylabel("Pressure [Pa]")
    axes[1].set_xlabel("Relative Time [$\mu$s]") 
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 4. ゲート内波形（重ね合わせ表示 / 2軸プロット）
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左軸: 実験データ (オレンジ)
    color_exp = 'tab:orange'
    ax1.set_xlabel("Relative Time [$\mu$s]") 
    # [単位変更] Voltage [V]
    ax1.set_ylabel("Voltage [V]", color=color_exp)
    line1, = ax1.plot(t_rel_exp, seg_exp, color=color_exp, label="Experiment Defect")
    ax1.tick_params(axis='y', labelcolor=color_exp)
    ax1.grid(True)

    # 右軸: シミュレーションデータ (青)
    ax2 = ax1.twinx()
    color_sim = 'tab:blue'
    # [単位変更] Pressure [Pa]
    ax2.set_ylabel("Pressure [Pa]", color=color_sim)
    line2, = ax2.plot(t_rel_sim, seg_sim, color=color_sim, linestyle='-', label="Simulation Defect")
    ax2.tick_params(axis='y', labelcolor=color_sim)

    # 凡例をまとめて表示
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title("Gated Waveform (Aligned)")
    plt.tight_layout()
    plt.show()


    # FFT計算
    fft_exp, freq = make_fftdata(seg_exp, dt_exp)
    fft_sim, _    = make_fftdata(seg_sim, dt_exp)
    fft_exp_smooth, _ = make_fftdata(seg_exp_smooth, dt_exp)
    fft_sim_smooth, _ = make_fftdata(seg_sim_smooth, dt_exp)

    # 5. 生スペクトル 0-10MHz (FFT.pyと同じ形式: 1軸・重ね描き)
    band_full = (freq > 0) & (freq <= 10e6)
    f_full = freq[band_full] / 1e6
    
    plt.figure(figsize=(8, 5))
    # 実験: オレンジ
    plt.plot(f_full, fft_exp[band_full], label="Exp Defect", color='tab:orange')
    # シミュレーション: 青
    plt.plot(f_full, fft_sim[band_full], label="Sim Defect", color='tab:blue')
    
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude")
    plt.title("Amplitude spectrum (0-10 MHz, gated)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# バンド抽出 (評価指標や正規化グラフ用には指定範囲 2-8MHz を使用)
    band = (freq >= freq_min) & (freq <= freq_max)
    f_axis = freq[band] / 1e6
    f_exp_tgt = fft_exp[band]
    f_sim_tgt = fft_sim[band]

    # 6. 単体スペクトルの正規化グラフ (FFT.pyにあったもの)
    fft_exp_norm_single = f_exp_tgt / np.max(f_exp_tgt)
    fft_sim_norm_single = f_sim_tgt / np.max(f_sim_tgt)

    plt.figure(figsize=(8, 5))
    plt.plot(f_axis, fft_exp_norm_single, label="Exp (Norm)", color='tab:orange')
    plt.plot(f_axis, fft_sim_norm_single, label="Sim (Norm)", color='tab:blue')
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Normalized Amplitude spectrum")
    plt.title(f"Normalized Amplitude spectrum ({freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------
    # 比の計算 & グラフ表示 (ここを修正: 0-10 MHz で表示)
    # ----------------------------------------------------------------
    
    # 1) 広帯域 (0-10 MHz) での比率計算 ※グラフ描画用
    # band_full は "5. 生スペクトル" の箇所で定義済みのものを使用
    ratio_exp_full = fft_exp[band_full] / (fft_exp_smooth[band_full] + 1e-12)
    ratio_sim_full = fft_sim[band_full] / (fft_sim_smooth[band_full] + 1e-12)

    # 7. 振幅比グラフ (Amplitude spectrum (Defect/Smooth)) [0-10 MHz]
    plt.figure(figsize=(8, 5))
    plt.plot(f_full, ratio_exp_full, label="Exp ", color='tab:orange')
    plt.plot(f_full, ratio_sim_full, label="Sim ", color='tab:blue')
    plt.title("Amplitude spectrum (Defect/Smooth) (0-10 MHz)") # タイトルを変更
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude spectrum (Defect/Smooth)")
    # 見やすいようにy軸範囲を制限したい場合は以下をコメントアウト解除
    # plt.ylim(0, 5) 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) 指定範囲 (2-8 MHz) での比率計算 ※後続の「正規化」および「評価指標」用
    ratio_exp = f_exp_tgt / (fft_exp_smooth[band] + 1e-12)
    ratio_sim = f_sim_tgt / (fft_sim_smooth[band] + 1e-12)

    # 正規化 (ここは指定範囲内での最大値で正規化するため ratio_exp を使用)
    norm_ratio_exp = ratio_exp / ratio_exp.max()
    norm_ratio_sim = ratio_sim / ratio_sim.max()

    # 8. 指定範囲 正規化 振幅比
    plt.figure(figsize=(8, 5))
    plt.plot(f_axis, norm_ratio_exp, label="Exp (Max=1)", color='tab:orange')
    plt.plot(f_axis, norm_ratio_sim, label="Sim (Max=1)", color='tab:blue')
    plt.title(f"Normalized Amplitude spectrum (Defect/Smooth) ({freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Normalized Amplitude spectrum (Defect/Smooth)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # ============================================================
    #  評価指標出力 (日本語フォーマット)
    # ============================================================
    print("\n" + "="*60)
    print(f"一致度評価 (範囲: {freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz)")
    print("※ MAE   : 0に近いほど全体的な誤差が小さい")
    print("※ RMSE  : 0に近いほど局所的な大きなズレが小さい")
    print("※ CosSim: 1に近いほど「形」が似ている (最大1.0)")
    print("-" * 60)
    
    w = 10
    
    print(f"--- 平滑化 (移動平均: window={w}) ---")
    print_metrics("Ratio Shape", smooth_data(norm_ratio_exp, w), smooth_data(norm_ratio_sim, w))
    
    print("-" * 60)
    print("--- 生データ (平滑化なし) ---")
    print_metrics("Ratio Shape", norm_ratio_exp, norm_ratio_sim)
    
    print("="*60 + "\n")

    # --------------------------------------------------------------------------------
    # 【追加】ゲート内波形の比較グラフ（平滑面 vs 欠陥あり）
    # --------------------------------------------------------------------------------
    
    # 3-1. 実験データ：平滑面(Smooth) と 欠陥(Defect) の比較
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True) 
    
    # 上段: 実験 平滑面 (グレー)
    axes[0].plot(t_rel_exp, seg_exp_smooth, color='gray', label="Experiment Smooth (Gated)")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Experiment: Smooth vs Defect (Gated)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # 下段: 実験 欠陥あり (オレンジ)
    axes[1].plot(t_rel_exp, seg_exp, color='tab:orange', label="Experiment Defect (Gated)")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Relative Time [$\mu$s]") 
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 3-2. シミュレーションデータ：平滑面(Smooth) と 欠陥(Defect) の比較
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True) 
    
    # 上段: シミュレーション 平滑面 (グレー)
    axes[0].plot(t_rel_sim, seg_sim_smooth, color='gray', label="Simulation Smooth (Gated)")
    axes[0].set_ylabel("Pressure [Pa]")
    axes[0].set_title("Simulation: Smooth vs Defect (Gated)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # 下段: シミュレーション 欠陥あり (青)
    axes[1].plot(t_rel_sim, seg_sim, color='tab:blue', label="Simulation Defect (Gated)")
    axes[1].set_ylabel("Pressure [Pa]")
    axes[1].set_xlabel("Relative Time [$\mu$s]") 
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# ================================================================================
    #  3. 論文スタイル比較グラフ (2軸独立版：軸色黒 & 凡例右上)
    # ================================================================================
    import matplotlib.patches as patches

    # --- ★配置調整パラメータ ---
    # (a) 実験データ (Voltage)
    exp_smooth_bottom_margin = 1.5  
    exp_rough_top_margin = 1.5      

    # (b) シミュレーションデータ (Pressure)
    sim_smooth_bottom_margin = 1.5
    sim_rough_top_margin = 1.5

    # "Tail wave" の開始時間 [s]
    tail_start_time = 0.8e-6 
    
    # --------------------------------------------------------

    # グラフ作成 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.4) 

    # 時間軸を [s] に変換
    t_sec_exp = t_rel_exp * 1e-6
    t_sec_sim = t_rel_sim * 1e-6

    # ==============================================================================
    # (a) Experiment
    # ==============================================================================
    ax1 = axes[0]          # 左軸
    ax2 = ax1.twinx()      # 右軸

    # --- プロット ---
    # Smooth (青)
    ln1 = ax1.plot(t_sec_exp, seg_exp_smooth, label="Smooth", color='tab:blue', linewidth=1.5)
    # Rough (オレンジ)
    ln2 = ax2.plot(t_sec_exp, seg_exp, label="Rough", color='orange', linewidth=1.5)

    # --- 軸ラベル (色は黒) ---
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Voltage [V]", color='black', fontsize=12)
    ax2.set_ylabel("Voltage [V]", color='black', fontsize=12)
    ax1.set_title("(a) Experiment", loc='left', fontsize=14, fontweight='bold')

    # --- 軸の色設定 (すべて黒) ---
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # --- 上下分離のための表示範囲調整 ---
    # Smooth(左軸)
    ymin1, ymax1 = seg_exp_smooth.min(), seg_exp_smooth.max()
    h1 = ymax1 - ymin1
    ax1.set_ylim(ymin1 - h1 * exp_smooth_bottom_margin, ymax1 + h1 * 0.1)

    # Rough(右軸)
    ymin2, ymax2 = seg_exp.min(), seg_exp.max()
    h2 = ymax2 - ymin2
    ax2.set_ylim(ymin2 - h2 * 0.1, ymax2 + h2 * exp_rough_top_margin)

    # --- ★凡例 (右上) ---
    lines_exp = ln1 + ln2
    labels_exp = [l.get_label() for l in lines_exp]
    ax1.legend(lines_exp, labels_exp, loc='upper right', fontsize=18)

    '''
    # --- Tail wave 枠 ---
    rect_h_exp = h2 * 0.6
    rect_y_exp = seg_exp.min()
    
    ax2.axvline(x=tail_start_time, color='red', linestyle='--', linewidth=1.5)
    
    rect_exp = patches.Rectangle((tail_start_time, rect_y_exp - rect_h_exp*0.2), 
                                 width=(t_sec_exp[-1] - tail_start_time), 
                                 height=rect_h_exp, 
                                 linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(rect_exp)
    ax2.text(tail_start_time + 0.2e-6, rect_y_exp + rect_h_exp, "Tail wave", 
             color='red', fontweight='bold', fontsize=10)
    
    '''

    # X軸指数表記
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


    # ==============================================================================
    # (b) Simulation
    # ==============================================================================
    ax3 = axes[1]          # 左軸
    ax4 = ax3.twinx()      # 右軸

    # --- プロット (正負を逆にするためデータに - を付与) ---
    # Smooth (青)
    ln3 = ax3.plot(t_sec_sim, -seg_sim_smooth, label="Smooth", color='tab:blue', linewidth=1.5)
    # Rough (オレンジ)
    ln4 = ax4.plot(t_sec_sim, -seg_sim, label="Rough", color='orange', linewidth=1.5)

    # --- 軸ラベル ---
    ax3.set_xlabel("Time [s]", fontsize=12)
    ax3.set_ylabel("Pressure [Pa]", color='black', fontsize=12)
    ax4.set_ylabel("Pressure [Pa]", color='black', fontsize=12)
    ax3.set_title("(b) Simulation", loc='left', fontsize=14, fontweight='bold')

    # --- 軸の色設定 ---
    ax3.tick_params(axis='y', labelcolor='black')
    ax4.tick_params(axis='y', labelcolor='black')

    # --- 上下分離のための表示範囲調整 (反転後の値で計算) ---
    # Smooth(左軸)
    seg_sim_smooth_inv = -seg_sim_smooth
    ymin3, ymax3 = seg_sim_smooth_inv.min(), seg_sim_smooth_inv.max()
    h3 = ymax3 - ymin3
    ax3.set_ylim(ymin3 - h3 * sim_smooth_bottom_margin, ymax3 + h3 * 0.1)

    # Rough(右軸)
    seg_sim_inv = -seg_sim
    ymin4, ymax4 = seg_sim_inv.min(), seg_sim_inv.max()
    h4 = ymax4 - ymin4
    ax4.set_ylim(ymin4 - h4 * 0.1, ymax4 + h4 * sim_rough_top_margin)

    # --- 凡例 (右上 / サイズを大きく設定) ---
    lines_sim = ln3 + ln4
    labels_sim = [l.get_label() for l in lines_sim]
    ax3.legend(lines_sim, labels_sim, loc='upper right', fontsize=18)

    '''
    # --- Tail wave 枠 ---
    rect_h_sim = h4 * 0.6
    rect_y_sim = seg_sim.min()

    ax4.axvline(x=tail_start_time, color='red', linestyle='--', linewidth=1.5)

    rect_sim = patches.Rectangle((tail_start_time, rect_y_sim - rect_h_sim*0.2), 
                                 width=(t_sec_sim[-1] - tail_start_time), 
                                 height=rect_h_sim, 
                                 linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
    ax4.add_patch(rect_sim)
    ax4.text(tail_start_time + 0.2e-6, rect_y_sim + rect_h_sim, "Tail wave", 
             color='red', fontweight='bold', fontsize=10)
    '''

    # X軸指数表記
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.show()
