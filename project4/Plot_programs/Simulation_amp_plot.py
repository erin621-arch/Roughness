import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import rfft, rfftfreq

# ============================================================
#  基本パラメータ
# ============================================================

fft_N = 2 ** 14

x_length = 0.02       # [m]
mesh_length = 1.0e-5  # [m]

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P波速度
dx = x_length / int(x_length / mesh_length)

# FDTD dt (シミュレーションのタイムステップ)
dt_sim = dx / cl / np.sqrt(6)
print(f"FDTD dt_sim = {dt_sim:.3e} [s]")

# ゲート幅（ピーク周り切り出し）
left  = 2 ** 10
right = (2 ** 10) * 3
sim_gate_start = 7560  # シミュレーション波形のゲート開始位置目安

# 解析周波数帯域
freq_min = 2.0e6
freq_max = 8.0e6

# ============================================================
#  ユーティリティ
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
        data = data[:n]
    
    howmanyzero = n - len(data)
    left_pad = howmanyzero // 2
    right_pad = howmanyzero - left_pad
    data_fft = np.pad(data, (left_pad, right_pad), mode="constant")
    
    X = rfft(data_fft)
    freqs = rfftfreq(n, d=dt_sample)
    magnitude = np.abs(X) / (n / 2)
    return magnitude, freqs

def get_sim_filepath(base_root, shape, pitch, depth_val):
    """シミュレーションファイルのパス生成"""
    # フォルダ名のマッピング
    shape_map = {
        "sankaku": "Sankaku",
        "kusabi":  "Kusabi",
        "ushape":   "Ushape",
        "smooth":  "Smooth",
        "kusabi2": "Kusabi" 
    }
    
    folder_key = shape
    if "kusabi" in shape.lower(): folder_key = "kusabi"
    
    dir_shape = shape_map.get(folder_key, shape.capitalize())
    sim_dir = os.path.join(base_root, dir_shape)

    # ピッチの数値化 (1.25 -> "125")
    p_str = str(int(pitch * 100)) if not pitch.is_integer() else str(int(pitch*100))
    
    # 深さの数値化 (0.20 -> "20", 0.125 -> "12.5")
    d_num = depth_val * 100
    if d_num.is_integer():
        d_str = str(int(d_num))
    else:
        d_str = str(d_num)

    # ファイル名生成: {shape}_cupy_pitch{p}_depth{d}.csv
    filename = f"{shape}_cupy_pitch{p_str}_depth{d_str}.csv"
    return os.path.join(sim_dir, filename)

# ============================================================
#  メイン処理
# ============================================================

if __name__ == "__main__":

    # --- ディレクトリ設定 ---
    # doc_path = r"C:/Users/hisay/OneDrive/ドキュメント/Test_folder"
    # doc_path = "/Users/hisayoshi/project_python/Roughness/Test_folder"
    doc_path = r"C:\Users\cs16\Roughness\project4"  # 研究室PC
    
    sim_base_dir = os.path.join(doc_path, "Simulation_Data")

    # --- 解析条件 ---
    target_shape_prefix = "ushape"  # ファイル名の先頭
    target_pitch_val = 1.25
    
    # ★ 比較したい深さのリスト
    # target_depth_list = [0.10,0.15,0.20] 
    target_depth_list = [0.125,0.15,0.20] 
    
    # 平滑面(Smooth)の設定
    smooth_pitch_val = 1.25
    smooth_depth_val = 0.0

    print("-" * 60)
    print("Simulation Multi-Depth Analysis")
    print(f"Shape: {target_shape_prefix}, Pitch: {target_pitch_val}")

    # 1. Smoothデータの読み込み (基準)
    # パス生成
    smooth_path = get_sim_filepath(sim_base_dir, "Smooth", smooth_pitch_val, smooth_depth_val)
    # ファイル名が特殊な場合のフォールバック
    if not os.path.exists(smooth_path):
        fname_alt = f"kukei_ori_cupy_pitch{int(smooth_pitch_val*100)}_depth0.csv"
        smooth_path = os.path.join(sim_base_dir, "Smooth", fname_alt)
    
    if not os.path.exists(smooth_path):
        print(f"Error: Smooth file not found at {smooth_path}")
        exit()

    print(f"Reading Smooth: {os.path.basename(smooth_path)}")
    wave_smooth = np.loadtxt(smooth_path, delimiter=",", dtype=float)

    # ゲート処理 & FFT
    tail_smooth = wave_smooth[sim_gate_start:]
    seg_smooth = kiritori2_safe(tail_smooth, left, right)
    fft_smooth, freq_axis = make_fftdata(seg_smooth, dt_sim)

    # グラフ準備
    plt.figure(figsize=(10, 6))
    # カラーマップ設定 (実験データに合わせるなら jet 等)
    colors = plt.cm.jet(np.linspace(0, 1, len(target_depth_list)))

    # 2. 各Depthデータの処理
    for i, depth in enumerate(target_depth_list):
        # パス生成
        target_path = get_sim_filepath(sim_base_dir, target_shape_prefix, target_pitch_val, depth)
        print(f"Processing Depth {depth}: {os.path.basename(target_path)}")

        if not os.path.exists(target_path):
            print(f"  Warning: File not found {target_path}")
            continue

        # 読み込み
        wave_sim = np.loadtxt(target_path, delimiter=",", dtype=float)

        # ゲート処理 & FFT
        tail_sim = wave_sim[sim_gate_start:]
        seg_sim = kiritori2_safe(tail_sim, left, right)
        fft_sim, _ = make_fftdata(seg_sim, dt_sim)

        # 比の計算 (Defect / Smooth)
        ratio = fft_sim / (fft_smooth + 1e-12)

        # バンド抽出 (2-8 MHz)
        band_mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
        f_band = freq_axis[band_mask] / 1e6 # MHz
        ratio_band = ratio[band_mask]

        # 正規化 (Max=1)
        if np.max(ratio_band) > 0:
            ratio_norm = ratio_band / np.max(ratio_band)
        else:
            ratio_norm = ratio_band

        # ★★★ ラベル表記の変更箇所 ★★★
        # depth 0.100m のような表記に変更 (小数点以下3桁固定)
        label_str = f"depth {depth:.3f}m" 

        # プロット
        plt.plot(f_band, ratio_norm, label=label_str, color=colors[i], linewidth=2)

    # --- グラフのスタイル設定 (Experiment_amp_plot.py 準拠) ---
    plt.xlabel("Frequency [MHz]", fontsize=18)
    plt.ylabel("Normalized Amplitude", fontsize=18)
    
    # 軸メモリのサイズを大きく
    plt.tick_params(labelsize=18)
    
    # 縦軸の最小値を0.2に固定
    plt.ylim(bottom=0.2)
    
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    
    print("表示します...")
    plt.show()