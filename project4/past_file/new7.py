##########################################################################
# 実験データ(平滑面 & 傷あり) とシミュレーション CSV を
#  「平滑面」と「傷あり」の両方について処理して、
#  FFT の比 R(f) = |FFT_defect| / |FFT_smooth| を
#  実験・シミュレーションで比較するスクリプト
#
#  ・実験: CSV 複数本を読み込み → ヘッダ2行スキップ → NaN行削除
#          → 各トレースごとに mode(最頻値) を引いてベースライン除去
#          → 平滑面/傷ありそれぞれで全トレース平均波形を作る
#  ・シミュ: FDTD の dt_sim から dt_exp に再補間（平滑/傷あり 共通）
#  ・実験用 exp_a2、シミュ用 sim_a2 を指定してゲート
#  ・ゼロパディングFFT (fft_N) → 0〜10 MHz
#  ・2〜8 MHz の周波数帯で、傷あり / 平滑面 の比をプロット
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import pandas as pd

# ================== 物性値・FDTD パラメータ ==================

fft_N = 2 ** 14

x_length = 0.02   # x方向の長さ [m]
y_length = 0.04   # y方向の長さ [m]
mesh_length = 1.0e-5  # メッシュ長 [m]
nx = int(x_length / mesh_length)
ny = int(y_length / mesh_length)

dx = x_length / nx
dy = y_length / ny

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P波速度
ct = np.sqrt(G / rho)                                     # S波速度
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

# FDTD の時間刻み（シミュ側 dt）
dt_sim = dx / cl / np.sqrt(6)
print(f"FDTD dt_sim = {dt_sim:.3e} [s] (約 {1.0/dt_sim/1e6:.3f} Msps)")

# ゲート用
left  = 2 ** 10           # ピーク前に残す点数
right = (2 ** 10) * 3     # ピーク後に残す点数

# ---- ゲート開始インデックス（ここは自由に調整）----
#   今回は平滑面・傷ありで同じ exp_a2 / sim_a2 を使う前提
exp_a2 = 0       # 実験データの反射開始（最初の大きな波の少し前）
sim_a2 = 3780    # シミュレーションの反射開始（2発目エコーの少し前）

# ================== ユーティリティ関数 ==================

def kiritori2(data_sample, left, right):
    """
    波形の最大値まわりに [left, right] だけ切り出す。
    戻り値:
        切り出しデータ, ピーク位置index, 切り出し開始index, 切り出し終了index
    """
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = datamaxhere - left
    datamaxend   = datamaxhere + right
    return data_sample[datamaxstart:datamaxend], datamaxhere, datamaxstart, datamaxend


def make_fftdata(data, dt_sample):
    """
    data を fft_N 点にゼロパディング（前後に半分ずつ）して FFT。
    """
    n = fft_N

    if len(data) > n:
        raise ValueError(
            f"データ長 {len(data)} が fft_N={n} を超えています。fft_N を増やしてください。"
        )

    howmanyzero = n - len(data)
    data_fft = np.concatenate(
        [np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))],
        axis=0
    )

    if len(data_fft) != n:
        if len(data_fft) > n:
            data_fft = data_fft[:n]
        else:
            data_fft = np.concatenate(
                [data_fft, np.zeros(n - len(data_fft))],
                axis=0
            )

    X = np.fft.fft(data_fft)
    freqs = np.linspace(0, 1.0 / dt_sample, n)
    magnitude = np.abs(X) / (n / 2)

    return magnitude, freqs


def interpolate_to_target_dt(data_raw, dt_original, dt_target):
    """
    波形 data_raw (dt_original 刻み) を dt_target 刻みに再補間する（1D）。
    """
    t_original = np.arange(0, len(data_raw) * dt_original, dt_original)
    t_max = (len(data_raw) - 1) * dt_original
    t_new = np.arange(0, t_max, dt_target)

    func = interp1d(t_original, data_raw, kind='cubic')
    data_resampled = func(t_new)
    return data_resampled


def load_experiment_mean(base_dir, exp_files):
    """
    実験 CSV 群を読み込んで平均波形を返す（平滑面・傷ありどちらにも使用）。
    ・ヘッダ2行をスキップ
    ・NaN 行を除去
    ・各トレースから mode(最頻値) を引いてベースライン除去
    戻り値:
        time_exp : 時間軸（1D）
        mean_wave: 平均波形（1D）
        all_waves: shape = (n_files, n_points) の配列
    """
    waves = []
    time_exp = None

    for fname in exp_files:
        path = os.path.join(base_dir, fname)
        print("読み込み (実験):", path)

        data = np.genfromtxt(path, delimiter=",", skip_header=2)

        if data.ndim == 1:
            raise RuntimeError(f"{fname} が 1 列しかない形式です。")

        # NaN行を削除
        data = data[~np.isnan(data).any(axis=1)]
        t = data[:, 0]
        v = data[:, 1]

        if time_exp is None:
            time_exp = t
        else:
            if len(t) != len(time_exp):
                print("Warning: 時間軸の長さが違います。", fname)

        # 「最頻値」を引いてベースライン除去
        mode_val = pd.Series(v).mode().iloc[0]
        v_detrended = v - mode_val

        waves.append(v_detrended)

    waves = np.vstack(waves)
    mean_wave = waves.mean(axis=0)

    return time_exp, mean_wave, waves


# ================== メイン処理 ==================

if __name__ == "__main__":

    base_dir = r"C:\Users\cs16\Documents\Test_folder\tmp_output"

    # --- シミュレーション CSV  ---
    # ★ここを自分のファイル名に合わせて書き換えてください
    #   例：平滑面 depth0 と 三角きず depth20
    file_sim_smooth = os.path.join(base_dir, "kukei1_ori_cupy_pitch125_depth0.csv")
    file_sim_defect = os.path.join(base_dir, "sankaku2_cupy_pitch125_depth10.csv")

    # --- 実験 CSV（平滑面 & 傷あり） ---
    # ★ここを自分の scope_??.csv に合わせて埋めてください

    # 平滑面（同じピッチの基準データ）
    exp_files_smooth = [
        "scope_106.csv",
        "scope_107.csv",
        "scope_108.csv",
        "scope_109.csv",
        "scope_110.csv",
        "scope_111.csv",
        "scope_112.csv",
        "scope_113.csv",
        "scope_114.csv",
        "scope_115.csv",
    ]

    # sankakuきずの場合 (Pitch125_Depth10)
    exp_files_defect = [
        "scope_52.csv", "scope_53.csv", "scope_54.csv", "scope_55.csv", "scope_56.csv",
        "scope_57.csv", "scope_58.csv", "scope_59.csv", "scope_60.csv", "scope_61.csv",
        "scope_62.csv", "scope_63.csv", "scope_64.csv", "scope_65.csv", "scope_66.csv",
        "scope_67.csv", "scope_68.csv", "scope_69.csv", "scope_70.csv", "scope_71.csv",
        "scope_72.csv", "scope_73.csv", "scope_74.csv", "scope_75.csv", "scope_76.csv",
    ]

    '''

    # 三角きず (125_20) の場合
    exp_files_defect = [
        "scope_77.csv",
        "scope_78.csv",
        "scope_79.csv",
        "scope_80.csv",
        "scope_81.csv",
        "scope_96.csv",
        "scope_97.csv",
        "scope_98.csv",
        "scope_99.csv",
        "scope_100.csv",
        "scope_101.csv",
        "scope_102.csv",
        "scope_103.csv",
        "scope_104.csv",
        "scope_105.csv",
    ]
    '''

    print("シミュレーション CSV (smooth) :", file_sim_smooth)
    print("シミュレーション CSV (defect) :", file_sim_defect)
    print("実験 CSV (smooth)              :", exp_files_smooth)
    print("実験 CSV (defect)              :", exp_files_defect)

    # ---------- シミュレーション波形（平滑面 & 傷あり） ----------
    wave_sim_smooth = np.loadtxt(file_sim_smooth, delimiter=",", dtype=float)
    wave_sim_defect = np.loadtxt(file_sim_defect, delimiter=",", dtype=float)
    print(f"len(wave_sim_smooth) = {len(wave_sim_smooth)}")
    print(f"len(wave_sim_defect) = {len(wave_sim_defect)}")

    # ---------- 実験波形（平滑面 & 傷あり） ----------
    time_exp_smooth, wave_exp_smooth_mean, _ = load_experiment_mean(base_dir, exp_files_smooth)
    time_exp_defect, wave_exp_defect_mean, _ = load_experiment_mean(base_dir, exp_files_defect)

    print(f"len(time_exp_smooth)      = {len(time_exp_smooth)}")
    print(f"len(wave_exp_smooth_mean) = {len(wave_exp_smooth_mean)}")
    print(f"len(time_exp_defect)      = {len(time_exp_defect)}")
    print(f"len(wave_exp_defect_mean) = {len(wave_exp_defect_mean)}")

    # dt は平滑面の時間軸から取得（傷あり側も基本同じはず）
    dt_exp = time_exp_smooth[1] - time_exp_smooth[0]
    print(f"推定 dt_exp = {dt_exp:.3e} [s]")

    # ================== シミュ波形を実験刻みに再補間 ==================

    wave_sim_smooth_resampled = interpolate_to_target_dt(wave_sim_smooth, dt_sim, dt_exp)
    wave_sim_defect_resampled = interpolate_to_target_dt(wave_sim_defect, dt_sim, dt_exp)
    print(f"len(wave_sim_smooth_resampled) = {len(wave_sim_smooth_resampled)}")
    print(f"len(wave_sim_defect_resampled) = {len(wave_sim_defect_resampled)}")

    # ================== 反射波部分のゲート ==================

    # --- 実験（平滑面） ---
    tail_exp_smooth = wave_exp_smooth_mean[exp_a2:]
    seg_exp_smooth, peak_idx_local_exp_s, seg_start_local_exp_s, seg_end_local_exp_s = kiritori2(
        tail_exp_smooth, left, right
    )
    peak_idx_global_exp_s  = exp_a2 + peak_idx_local_exp_s
    seg_start_global_exp_s = exp_a2 + seg_start_local_exp_s
    seg_end_global_exp_s   = exp_a2 + seg_end_local_exp_s

    # --- 実験（傷あり） ---
    tail_exp_defect = wave_exp_defect_mean[exp_a2:]
    seg_exp_defect, peak_idx_local_exp_d, seg_start_local_exp_d, seg_end_local_exp_d = kiritori2(
        tail_exp_defect, left, right
    )
    peak_idx_global_exp_d  = exp_a2 + peak_idx_local_exp_d
    seg_start_global_exp_d = exp_a2 + seg_start_local_exp_d
    seg_end_global_exp_d   = exp_a2 + seg_end_local_exp_d

    # --- シミュ（平滑面） ---
    tail_sim_smooth = wave_sim_smooth_resampled[sim_a2:]
    seg_sim_smooth, peak_idx_local_sim_s, seg_start_local_sim_s, seg_end_local_sim_s = kiritori2(
        tail_sim_smooth, left, right
    )
    peak_idx_global_sim_s  = sim_a2 + peak_idx_local_sim_s
    seg_start_global_sim_s = sim_a2 + seg_start_local_sim_s
    seg_end_global_sim_s   = sim_a2 + seg_end_local_sim_s

    # --- シミュ（傷あり） ---
    tail_sim_defect = wave_sim_defect_resampled[sim_a2:]
    seg_sim_defect, peak_idx_local_sim_d, seg_start_local_sim_d, seg_end_local_sim_d = kiritori2(
        tail_sim_defect, left, right
    )
    peak_idx_global_sim_d  = sim_a2 + peak_idx_local_sim_d
    seg_start_global_sim_d = sim_a2 + seg_start_local_sim_d
    seg_end_global_sim_d   = sim_a2 + seg_end_local_sim_d

    print("=== ゲート情報（実験）===")
    print(f"[EXP smooth] peak index (global) = {peak_idx_global_exp_s}")
    print(f"[EXP smooth] gate range (global) = [{seg_start_global_exp_s}, {seg_end_global_exp_s})")
    print(f"[EXP defect] peak index (global) = {peak_idx_global_exp_d}")
    print(f"[EXP defect] gate range (global) = [{seg_start_global_exp_d}, {seg_end_global_exp_d})")

    print("=== ゲート情報（シミュ）===")
    print(f"[SIM smooth] peak index (global) = {peak_idx_global_sim_s}")
    print(f"[SIM smooth] gate range (global) = [{seg_start_global_sim_s}, {seg_end_global_sim_s})")
    print(f"[SIM defect] peak index (global) = {peak_idx_global_sim_d}")
    print(f"[SIM defect] gate range (global) = [{seg_start_global_sim_d}, {seg_end_global_sim_d})")

    # ================== プロット0：実験平均波形＋ゲート確認 ==================

    plt.figure(figsize=(10, 4))
    idx_exp = np.arange(len(wave_exp_smooth_mean))
    plt.plot(idx_exp, wave_exp_smooth_mean, label="experiment smooth (mean)")
    plt.plot(idx_exp, wave_exp_defect_mean, label="experiment defect (mean)", alpha=0.7)

    plt.axvline(exp_a2, color="g", linestyle=":", label=f"exp_a2={exp_a2}")
    plt.axvspan(seg_start_global_exp_s, seg_end_global_exp_s, color="orange", alpha=0.3,
                label="exp gate (smooth)")
    plt.axvspan(seg_start_global_exp_d, seg_end_global_exp_d, color="red", alpha=0.15,
                label="exp gate (defect)")

    plt.xlabel("Index")
    plt.ylabel("Amplitude [V]")
    plt.title("Experimental waveforms (smooth & defect) and gated reflection regions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== プロット1：シミュ波形＋ゲート確認 ==================

    plt.figure(figsize=(10, 4))
    idx_sim = np.arange(len(wave_sim_smooth_resampled))
    plt.plot(idx_sim, wave_sim_smooth_resampled, label="simulation smooth (resampled)")
    plt.plot(idx_sim, wave_sim_defect_resampled, label="simulation defect (resampled)", alpha=0.7)

    plt.axvline(sim_a2, color="g", linestyle=":", label=f"sim_a2={sim_a2}")
    plt.axvspan(seg_start_global_sim_s, seg_end_global_sim_s, color="orange", alpha=0.3,
                label="sim gate (smooth)")
    plt.axvspan(seg_start_global_sim_d, seg_end_global_sim_d, color="red", alpha=0.15,
                label="sim gate (defect)")

    plt.xlabel("Index")
    plt.ylabel("Amplitude [arb.]")
    plt.title("Simulation waveforms (smooth & defect, resampled) and gated regions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== FFT（どちらも dt_exp 基準） ==================

    fft_exp_smooth, freq = make_fftdata(seg_exp_smooth, dt_exp)
    fft_exp_defect, _    = make_fftdata(seg_exp_defect, dt_exp)
    fft_sim_smooth, _    = make_fftdata(seg_sim_smooth, dt_exp)
    fft_sim_defect, _    = make_fftdata(seg_sim_defect, dt_exp)

    band_2_8 = (freq >= 2e6) & (freq <= 8e6)
    freq_2_8_MHz = freq[band_2_8] / 1e6

    fft_exp_smooth_2_8 = fft_exp_smooth[band_2_8]
    fft_exp_defect_2_8 = fft_exp_defect[band_2_8]
    fft_sim_smooth_2_8 = fft_sim_smooth[band_2_8]
    fft_sim_defect_2_8 = fft_sim_defect[band_2_8]

    # ================== 傷あり / 平滑面 の比 ==================

    eps = 1e-12  # ゼロ割り防止用

    ratio_exp = fft_exp_defect_2_8 / (fft_exp_smooth_2_8 + eps)
    ratio_sim = fft_sim_defect_2_8 / (fft_sim_smooth_2_8 + eps)

    plt.figure(figsize=(8, 5))
    plt.plot(freq_2_8_MHz, ratio_exp, label="experiment: defect / smooth")
    plt.plot(freq_2_8_MHz, ratio_sim, label="simulation: defect / smooth")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude ratio (defect / smooth)")
    plt.title("Amplitude ratio spectrum (defect / smooth, 2–8 MHz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("平滑面 vs 傷あり の FFT 比 (実験 & シミュ) 比較終了")
