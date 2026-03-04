import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.fft import rfft, rfftfreq

# ============================================================
#  基本パラメータ
# ============================================================
fft_N = 2 ** 14
dt_default = 1.0e-8 # 万が一取得できない場合のデフォルト

# ゲート幅（ピーク周り切り出し）
left  = 2 ** 10
right = (2 ** 10) * 3
gate_start_idx = 0 # 実験データはピーク位置基準で切り出すため0または調整

# 解析周波数帯域
freq_min = 2.0e6
freq_max = 8.0e6

# ============================================================
#  ユーティリティ関数
# ============================================================

def kiritori2_safe(data_sample, left, right):
    """波形の最大値まわりに切り出し"""
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend]

def make_fftdata(data, dt_sample):
    """FFT計算"""
    n = fft_N
    if len(data) > n:
        # データが長すぎる場合はクリップするかエラーにする
        data = data[:n]
        
    howmanyzero = n - len(data)
    left_pad = howmanyzero // 2
    right_pad = howmanyzero - left_pad
    data_fft = np.pad(data, (left_pad, right_pad), mode="constant")
    
    X = rfft(data_fft)
    freqs = rfftfreq(n, d=dt_sample)
    magnitude = np.abs(X) / (n / 2)
    return magnitude, freqs

def load_experiment_from_folder(target_dir):
    """指定フォルダ内のCSVを読み込んで平均化"""
    waves = []
    time_exp = None
    
    if not os.path.exists(target_dir):
        return None, None
    
    search_path = os.path.join(target_dir, "scope_*.csv")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        return None, None

    # print(f"  Reading {len(file_list)} files from: {os.path.basename(target_dir)}")

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
            
        except:
            continue

    if not waves:
        return None, None

    min_len = min(len(w) for w in waves)
    if len(time_exp) > min_len:
        time_exp = time_exp[:min_len]

    waves_trimmed = [w[:min_len] for w in waves]
    mean_wave = np.mean(waves_trimmed, axis=0)
    
    return time_exp, mean_wave

def get_exp_folder_path(base_root, shape, pitch, depth):
    """フォルダパス生成"""
    if shape.lower() == "smooth":
        return os.path.join(base_root, "Smooth", "0_0")

    shape_map = {"sankaku": "Sankaku", "kusabi": "Kusabi", "hanen": "Hanen"}
    dir_shape = shape_map.get(shape, shape.capitalize())
    
    p_val = pitch.replace("p", "").replace(".", "")
    
    try:
        d_num = float(depth.replace("d", ""))
        val_100 = d_num * 100
        if val_100.is_integer():
            d_str = str(int(val_100))
        else:
            d_str = str(val_100)
    except:
        d_str = depth.replace("d", "").replace(".", "")

    dir_name = f"{p_val}_{d_str}"
    return os.path.join(base_root, dir_shape, dir_name)

# ============================================================
#  メイン処理
# ============================================================

if __name__ == "__main__":

    # --- パス設定 (環境に合わせて変更してください) ---
    doc_path = r"C:/Users/hisay/OneDrive/ドキュメント/Test_folder"
    exp_base_dir = os.path.join(doc_path, "Experiment_Data")

    # --- 解析条件 ---
    target_shape = "Hanen"   # "sankaku" / "kusabi" / "hanen"
    target_pitch = "p1.25"
    
    # ★比較したい深さのリスト 
    # depth_list = ["d0.10", "d0.15", "d0.20"]
    depth_list = ["d0.125", "d0.15", "d0.20"] 

    print("-" * 60)
    print("Multi-Depth Experiment Spectrum Ratio Analysis")
    print(f"Shape: {target_shape}, Pitch: {target_pitch}")
    print(f"Depths: {depth_list}")

    # 1. Smoothデータの読み込み (基準)
    smooth_dir = get_exp_folder_path(exp_base_dir, "smooth", "", "")
    t_smooth, wave_smooth = load_experiment_from_folder(smooth_dir)
    
    if wave_smooth is None:
        print("Error: Smooth data not found.")
        exit()

    dt_exp = t_smooth[1] - t_smooth[0]
    
    # ゲート切り出し & FFT
    seg_smooth = kiritori2_safe(wave_smooth[gate_start_idx:], left, right)
    fft_smooth, freq = make_fftdata(seg_smooth, dt_exp)

    # バンド抽出用マスク
    band_mask = (freq >= freq_min) & (freq <= freq_max)
    f_axis = freq[band_mask] / 1e6  # MHz表記用

    # グラフ準備
    plt.figure(figsize=(10, 6))
    colors = plt.cm.jet(np.linspace(0, 1, len(depth_list)))

    # 2. 各Depthデータの処理
    for i, d_val in enumerate(depth_list):
        print(f"Processing: {d_val} ...")
        
        target_dir = get_exp_folder_path(exp_base_dir, target_shape, target_pitch, d_val)
        _, wave_defect = load_experiment_from_folder(target_dir)
        
        if wave_defect is not None:
            # ゲート切り出し & FFT
            seg_defect = kiritori2_safe(wave_defect[gate_start_idx:], left, right)
            fft_defect, _ = make_fftdata(seg_defect, dt_exp)
            
            # --- 比の計算 (Defect / Smooth) ---
            # 指定帯域のみ抽出
            spec_smooth_band = fft_smooth[band_mask]
            spec_defect_band = fft_defect[band_mask]
            
            # ゼロ除算防止
            ratio = spec_defect_band / (spec_smooth_band + 1e-12)
            
            # --- 正規化 (Max=1) ---
            norm_ratio = ratio / np.max(ratio)
            
            # ラベル整形
            try:
                val = float(d_val.replace("d", ""))
                label_str = f"depth {val:.3f}mm"
            except:
                label_str = d_val
            
            # プロット
            plt.plot(f_axis, norm_ratio, label=label_str, color=colors[i], linewidth=2)
            
        else:
            print(f"  Warning: Data not found for {d_val}")

    # グラフ仕上げ
    # plt.title(f"[Experiment] Normalized (Defect/Smooth)\n{target_shape} {target_pitch} ({freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz)", fontsize=14)
    plt.xlabel("Frequency [MHz]", fontsize=18)
    plt.ylabel("Normalized Amplitude", fontsize=18)

    # ★★★ ここを追加: 軸メモリ（数字）のサイズを大きくする ★★★
    plt.tick_params(labelsize=18)
    
    # ★ 縦軸の最小値を0.5に固定
    plt.ylim(bottom=0.2)
    
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    
    print("表示します...")
    plt.show()