##########################################################################
#  実験（平滑面10本 + きずあり10本）と
#  シミュレーション（平滑面 + きずあり）の FFT 比スペクトルを比較するスクリプト
#
#  ・実験: 各トレースごとに mode を引いてベースライン除去 → 10本平均
#          exp_a2 以降を反射としてゲート → FFT → 2〜8 MHz 抽出
#          -> |FFT(defect)| / |FFT(smooth)| を計算
#  ・シミュ: depth0 / depth>0 の CSV を dt_exp に再補間
#            sim_a2 以降を反射としてゲート → FFT → 2〜8 MHz
#            -> |FFT(defect)| / |FFT(smooth)| を計算
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# ================== 共通パラメータ ==================

fft_N = 2 ** 14
left  = 2 ** 10           # ピーク前
right = (2 ** 10) * 3     # ピーク後

# --- FDTD 側パラメータ（dt_sim を再計算） ---
x_length = 0.02
y_length = 0.04
mesh_length = 1.0e-5
nx = int(x_length / mesh_length)
dx = x_length / nx

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))
dt_sim = dx / cl / np.sqrt(6)
print(f"FDTD dt_sim = {dt_sim:.3e} [s]")

# ---- ゲート開始インデックス（必要に応じて調整）----
exp_a2 = 1000   # 実験：反射開始
sim_a2 = 3780   # シミュ：反射開始（再補間後 index）

# ================== ユーティリティ関数 ==================

def kiritori2(data_sample, left, right):
    """最大値まわり [left, right] を切り出し"""
    idx_peak = np.nanargmax(data_sample)
    s = idx_peak - left
    e = idx_peak + right
    return data_sample[s:e], idx_peak, s, e

def make_fftdata(data, dt_sample):
    """ゼロパディング FFT"""
    n = fft_N
    if len(data) > n:
        raise ValueError(f"len(data)={len(data)} > fft_N={n}")

    howmanyzero = n - len(data)
    pad_left = howmanyzero // 2
    pad_right = howmanyzero - pad_left
    data_fft = np.concatenate([np.zeros(pad_left), data, np.zeros(pad_right)])

    X = np.fft.fft(data_fft)
    freqs = np.linspace(0, 1.0 / dt_sample, n)
    mag = np.abs(X) / (n / 2)
    return mag, freqs

def extract_band(freq, spec, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    return spec[mask], freq[mask]

def interpolate_sim_one(data_raw, dt_original, dt_target):
    """シミュ波形を実験の dt に再補間"""
    t = np.arange(0, len(data_raw) * dt_original, dt_original)
    t_max = (len(data_raw) - 1) * dt_original
    t_new = np.arange(0, t_max, dt_target)
    func = interp1d(t, data_raw, kind="cubic")
    return func(t_new)

def remove_baseline_mode(w):
    """最頻値を引いてベースライン除去（sourse_new.py のノリ）"""
    vals, counts = np.unique(np.round(w, 6), return_counts=True)
    mode_val = vals[np.argmax(counts)]
    return w - mode_val

# ================== メイン ==================

if __name__ == "__main__":

   # base_dir = r"C:\Users\cs16\Documents\Test_folder\tmp_output"
    base_dir = r"C:\Users\hisay\OneDrive\ドキュメント\Test_folder\tmp_output"   # 自宅PC

    # ---- 実験ファイル名リスト（★ここを自分のファイルに合わせて編集）----
    # 例: 平滑面10本
    exp_smooth_files = [
        "scope_14.csv",
        "scope_15.csv",
        "scope_16.csv",
        "scope_17.csv",
        "scope_18.csv",
    ]

    # 例: 三角きず(p=1.25, d=10mm) の10本
    exp_defect_files = [
        "scope_86.csv",
        "scope_87.csv",
        "scope_88.csv",
        "scope_89.csv",
        "scope_90.csv",
    ]

    # ---- 実験：平滑面10本の平均波形 ----
    waves_smooth = []
    dt_exp = None

    for fname in exp_smooth_files:
        path = os.path.join(base_dir, fname)
        data = np.genfromtxt(path, delimiter=",", skip_header=2)
        data = data[~np.isnan(data).any(axis=1)]
        t = data[:, 0]
        w = data[:, 1]

        dt_here = t[1] - t[0]
        if dt_exp is None:
            dt_exp = dt_here

        w_bs = remove_baseline_mode(w)
        waves_smooth.append(w_bs)

    if dt_exp is None:
        raise RuntimeError("exp_smooth_files が空です。ファイル名を設定してください。")

    waves_smooth = np.array(waves_smooth)
    wave_exp_smooth = np.mean(waves_smooth, axis=0)
    print("平滑面実験: shape =", waves_smooth.shape)

    # ---- 実験：きずあり10本の平均波形 ----
    waves_defect = []

    for fname in exp_defect_files:
        path = os.path.join(base_dir, fname)
        data = np.genfromtxt(path, delimiter=",", skip_header=2)
        data = data[~np.isnan(data).any(axis=1)]
        t = data[:, 0]
        w = data[:, 1]

        # 念のため dt 一致チェック（ずれてたら警告だけ）
        dt_here = t[1] - t[0]
        if abs(dt_here - dt_exp) > 1e-12:
            print(f"Warning: {fname} の dt が少し違うかも: {dt_here:.3e}")

        w_bs = remove_baseline_mode(w)
        waves_defect.append(w_bs)

    waves_defect = np.array(waves_defect)
    wave_exp_defect = np.mean(waves_defect, axis=0)
    print("きずあり実験: shape =", waves_defect.shape)
    print(f"dt_exp = {dt_exp:.3e} s")

    # ---- シミュレーション（平滑面 & きずあり）----
    sim_smooth_file = "sankaku2_cupy_pitch125_depth0.csv"
    sim_def_file    = "kusabi_cupy_pitch125_depth20.csv"

    wave_sim_smooth = np.loadtxt(os.path.join(base_dir, sim_smooth_file),
                                 delimiter=",", dtype=float)
    wave_sim_defect = np.loadtxt(os.path.join(base_dir, sim_def_file),
                                 delimiter=",", dtype=float)

    # シミュ波形を実験の dt に再補間
    wave_sim_smooth_rs = interpolate_sim_one(wave_sim_smooth, dt_sim, dt_exp)
    wave_sim_defect_rs = interpolate_sim_one(wave_sim_defect, dt_sim, dt_exp)

    # ================== ゲート & FFT（実験） ==================

    tail_smooth_exp = wave_exp_smooth[exp_a2:]
    tail_def_exp    = wave_exp_defect[exp_a2:]

    seg_smooth_exp, *_ = kiritori2(tail_smooth_exp, left, right)
    seg_def_exp,    *_ = kiritori2(tail_def_exp,    left, right)

    fft_smooth_exp, freq = make_fftdata(seg_smooth_exp, dt_exp)
    fft_def_exp,   _     = make_fftdata(seg_def_exp,   dt_exp)

    band_smooth_exp, freq_band = extract_band(freq, fft_smooth_exp, 2e6, 8e6)
    band_def_exp,   _          = extract_band(freq, fft_def_exp,    2e6, 8e6)

    ratio_exp = band_def_exp / band_smooth_exp

    # ================== ゲート & FFT（シミュ） ==================

    tail_smooth_sim = wave_sim_smooth_rs[sim_a2:]
    tail_def_sim    = wave_sim_defect_rs[sim_a2:]

    seg_smooth_sim, *_ = kiritori2(tail_smooth_sim, left, right)
    seg_def_sim,    *_ = kiritori2(tail_def_sim,    left, right)

    fft_smooth_sim, freq2 = make_fftdata(seg_smooth_sim, dt_exp)
    fft_def_sim,    _     = make_fftdata(seg_def_sim,    dt_exp)

    band_smooth_sim, freq_band2 = extract_band(freq2, fft_smooth_sim, 2e6, 8e6)
    band_def_sim,   _           = extract_band(freq2, fft_def_sim,    2e6, 8e6)

    ratio_sim = band_def_sim / band_smooth_sim

    # ================== 比スペクトルの比較プロット ==================

    fMHz = freq_band / 1e6

    plt.figure(figsize=(8, 5))
    plt.plot(fMHz, ratio_exp, label="Experiment (defect/smooth)")
    plt.plot(fMHz, ratio_sim, label="Simulation (defect/smooth)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude ratio")
    plt.title("FFT ratio spectrum (2–8 MHz, defect / smooth)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("実験 vs シミュ の FFT 比スペクトル比較終了")
