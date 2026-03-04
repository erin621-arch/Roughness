##########################################################################
#  矩形きずシミュレーションCSVに対して
#    再補間 (exp_dt) → ピーク周り切り出し → ゼロパディングFFT
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# ================== FDTD と同じ物性値・dt を再計算 ==================

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

# FDTD の時間刻み
dt = dx / cl / np.sqrt(6)

print(f"FDTD dt = {dt:.3e} [s] (約 {1.0/dt/1e6:.3f} Msps)")

# ================== FFT 用パラメータ　 ==================

fft_N = 2 ** 14           # FFT 長（固定）
left  = 2 ** 10           # ピーク前に残す点数
right = (2 ** 10) * 3     # ピーク後に残す点数

# 実験データと合わせるための「目標サンプリング間隔」
exp_dt = 0.0005E-06       # [s] 

# 再補間後の波形について、
# 「このステップ以降を反射波とみなす」開始位置
t_offset_resampled = 8000

# ================== ユーティリティ関数（もとのコードと同じ仕様） ==================

def kiritori2(data_sample, left, right):
    """
    波形の最大値まわりに [left, right] だけ切り出す。
    戻り値:
        切り出しデータ, ピーク位置index, 切り出し開始index, 切り出し終了index
    （index は data_sample に対するローカルな値）
    """
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = datamaxhere - left
    datamaxend   = datamaxhere + right

    # 範囲外に出ないようにクリップ（安全策）
    datamaxstart = max(datamaxstart, 0)
    datamaxend   = min(datamaxend, len(data_sample))

    return data_sample[datamaxstart:datamaxend], datamaxhere, datamaxstart, datamaxend


def make_fftdata(data, dt_sample):
    """
    - data を fft_N 点にゼロパディング（前後に半分ずつ）
    - numpy.fft.fft を使用
    - 周波数軸は np.linspace(0, 1/dt_sample, fft_N)
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

    # 念のため n と data_fft 長を揃える
    if len(data_fft) != n:
        if len(data_fft) > n:
            data_fft = data_fft[:n]
        else:
            data_fft = np.concatenate(
                [data_fft, np.zeros(n - len(data_fft))],
                axis=0
            )

    # 複素 FFT（両側に出るが、周波数軸は 0〜1/dt_sample ）
    X = np.fft.fft(data_fft)
    freqs = np.linspace(0, 1.0 / dt_sample, n)

    magnitude = np.abs(X) / (n / 2)

    return magnitude, freqs


def interpolate_sim_one(data_raw):
    """
    シミュレーション波形を FDTD の dt から exp_dt に再補間する。

    """
    # 元の時間軸（FDTD の dt）
    sim_t = np.arange(0, len(data_raw) * dt, dt)
    # 再補間後の時間軸（exp_dt）
    sim_t_new = np.arange(0, (len(data_raw) - 1) * dt, exp_dt)

    func = interp1d(sim_t, data_raw, kind='cubic')
    data_resampled = func(sim_t_new)

    return data_resampled

# ================== メイン処理 ==================

# シミュレーションCSVの場所
base_dir = r"C:\Users\cs16\Documents\test_folder(20251120)\tmp_output"

# FDTDコードと同じ命名規則
# ★必要に応じて f_pitch, f_depth を変えてください
f_pitch = 1.75e-3   # [m]
f_depth = 0.10e-3   # [m]

file_kukei = os.path.join(
    base_dir,
    f"kukei_ori_cupy_pitch{int(f_pitch * 1e5)}_depth{int(f_depth * 1e5)}.csv"
)

print("読み込みファイル（矩形きずシミュレーション）:")
print("  ", file_kukei)

# ---- 波形読み込み（FDTD の dt でサンプリングされたデータ） ----
wave_kukei = np.loadtxt(file_kukei, delimiter=',', dtype=float)
print(f"len(wave_kukei) = {len(wave_kukei)}")

# ---- まず exp_dt で再補間 ----
wave_kukei_resampled = interpolate_sim_one(wave_kukei)
print(f"len(wave_kukei_resampled) = {len(wave_kukei_resampled)}")

# ---- t_offset_resampled 以降だけを「反射候補」として扱う ----
a1 = t_offset_resampled
wave_tail = wave_kukei_resampled[a1:]   # 送信波の山は無視して、その後ろだけを見る

# ---- ピーク周辺の切り出し（反射部分の中でピーク探索）---- 
seg_kukei, peak_idx_local, seg_start_local, seg_end_local = kiritori2(
    wave_tail, left, right
)

# 再補間後の波形に対するグローバルインデックスに戻す
peak_idx_global   = a1 + peak_idx_local
seg_start_global  = a1 + seg_start_local
seg_end_global    = a1 + seg_end_local

print(f"ピーク位置 (resampled index) = {peak_idx_global}")
print(f"切り出し区間 (resampled index): [{seg_start_global}, {seg_end_global})")
print(f"len(seg_kukei) = {len(seg_kukei)}")

# ★ここでは DC オフセットは引かない
seg_kukei_used = seg_kukei

# ---- FFT 実行（ make_fftdata と同じ手法） ----
fft_kukei, freq = make_fftdata(seg_kukei_used, exp_dt)

print(f"len(fft_kukei) = {len(fft_kukei)}")
print(f"len(freq)      = {len(freq)}")

# ================== プロット0：再補間後の時間波形 + t_offset + ゲート ==================

plt.figure(figsize=(10, 4))
time_index = np.arange(len(wave_kukei_resampled))
plt.plot(time_index, wave_kukei_resampled, label="resampled wave (kukei)")

# 送信波と反射の境目（解析開始）を縦線で表示
plt.axvline(a1, color="g", linestyle=":", label=f"t_offset_resampled={a1}")

# 反射部分のピーク
plt.axvline(peak_idx_global, color="r", linestyle="--", label="peak (in tail)")

# ゲート領域
plt.axvspan(seg_start_global, seg_end_global, color="orange", alpha=0.3,
            label="gate region (reflection)")

plt.xlabel("Resampled index")
plt.ylabel("Amplitude")
plt.title("Resampled waveform (kukei), t_offset and gated reflection region")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================== プロット1：切り出し後の時間波形 ==================

plt.figure(figsize=(10, 4))
t_seg = np.arange(len(seg_kukei_used)) * exp_dt
plt.plot(t_seg * 1e6, seg_kukei_used)
plt.xlabel("Time [µs]")
plt.ylabel("Amplitude")
plt.title("Gated waveform around reflected peak (kukei, resampled)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== プロット2：0〜10 MHz の FFT スペクトル ==================

band_full = (freq > 0) & (freq <= 10e6)
plt.figure(figsize=(8, 8))
plt.plot(freq[band_full] / 1e6, fft_kukei[band_full])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Amplitude")
plt.title("FFT spectrum (0–10 MHz, kukei, reflected part, resampled)")
plt.grid(True)
plt.xlim(0, 10)
plt.tight_layout()
plt.show()

# ================== プロット3：2〜8 MHz のみ（正規化して形を見る） ==================

band = (freq >= 2e6) & (freq <= 8e6)
freq_MHz = freq[band] / 1e6
amp_band = fft_kukei[band]
amp_norm = amp_band / np.max(amp_band)

plt.figure(figsize=(10, 4))
plt.plot(freq_MHz, amp_norm)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Normalized amplitude")
plt.title("Normalized FFT spectrum (2–8 MHz, kukei, reflected part, resampled)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================== ピーク周波数の表示 ==================

peak_idx_band = np.argmax(amp_band)
peak_freq = freq_MHz[peak_idx_band]
print(f"反射部分 2-8 MHz 帯域内のピーク周波数 ≈ {peak_freq:.3f} MHz")

print("矩形きずFFT解析終了")
