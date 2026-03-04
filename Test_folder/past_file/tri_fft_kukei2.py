##########################################################################
#  矩形きずシミュレーションCSVに対して
#    再補間 (exp_dt) → ピーク周り切り出し → ゼロパディングFFT
#    ＋ 基準波形（depth=0）で割ったスペクトルを算出する版
#    ＋ 平滑面(depth0)ときずあり(depth>0)のFFTを重ねて表示
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

# ================== ユーティリティ関数 ==================

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

    # 範囲外に出ないようにクリップ（必要なら有効化）
    # datamaxstart = max(datamaxstart, 0)
    # datamaxend   = min(datamaxend, len(data_sample))

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


def extract_frequency_band(freq, fft_data, freq_min, freq_max):
    """
    周波数配列 freq・スペクトル fft_data から、
    [freq_min, freq_max] の帯域だけを取り出す。
    """
    mask = (freq >= freq_min) & (freq <= freq_max)
    freq_band = freq[mask]
    fft_band  = fft_data[mask]
    return fft_band, freq_band


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


def make_data(data0_resampled, data_resampled, a1):
    """
    基準波形 data0_resampled と比較対象 data_resampled を使って、
    - a1 以降を取り出し
    - ピーク周りを kiritori2 で切り出し
    - ゼロパディング FFT
    - 2〜8 MHz 抽出
    - data / data0 の振幅スペクトル比を返す

    戻り値:
        ratio: 2〜8 MHz 帯域における |FFT(data) / FFT(data0)| （1D配列）
        freq_band: 対応する周波数配列 [Hz]
    """
    # a1 以降だけを見る（送信波は無視）
    tail      = data_resampled[a1:]
    tail0     = data0_resampled[a1:]

    # ピークまわりを [left, right] で切り出し
    wave,  _, _, _  = kiritori2(tail,  left, right)
    wave0, _, _, _  = kiritori2(tail0, left, right)

    # FFT（ゼロパディング込み）
    fft_data,  freq = make_fftdata(wave,  exp_dt)
    fft_data0, _    = make_fftdata(wave0, exp_dt)

    # 2〜8 MHz 帯域だけ取り出す
    band,  freq_band  = extract_frequency_band(freq, fft_data,  2e6, 8e6)
    band0, _          = extract_frequency_band(freq, fft_data0, 2e6, 8e6)

    # 基準波形で割ったスペクトル（振幅の比）
    ratio = np.abs(band / band0)

    return ratio, freq_band

# ================== メイン処理 ==================

if __name__ == "__main__":

    # シミュレーションCSVの場所
    base_dir = r"C:\\Users\\cs16\\Documents\\test_folder(20251120)\\tmp_output"

    # FDTDコードと同じ命名規則
    # ★必要に応じて f_pitch, f_depth を変えてください
    f_pitch = 1.25e-3   # [m]
    f_depth = 0.20e-3   # [m]

    # 対象きず波形
    file_kukei = os.path.join(
        base_dir,
        f"kukei_ori_cupy_pitch{int(f_pitch * 1e5)}_depth{int(f_depth * 1e5)}.csv"
    )

    # 基準波形（depth=0 を想定）
    file_ref = os.path.join(
        base_dir,
        f"kukei_ori_cupy_pitch{int(f_pitch * 1e5)}_depth0.csv"
    )

    print("読み込みファイル（矩形きずシミュレーション）:")
    print("  基準 :", file_ref)
    print("  対象 :", file_kukei)

    # ---- 波形読み込み（FDTD の dt でサンプリングされたデータ）----
    wave_ref   = np.loadtxt(file_ref,   delimiter=',', dtype=float)
    wave_kukei = np.loadtxt(file_kukei, delimiter=',', dtype=float)
    print(f"len(wave_ref)   = {len(wave_ref)}")
    print(f"len(wave_kukei) = {len(wave_kukei)}")

    # ---- まず exp_dt で再補間 ----
    wave_ref_resampled   = interpolate_sim_one(wave_ref)
    wave_kukei_resampled = interpolate_sim_one(wave_kukei)
    print(f"len(wave_ref_resampled)   = {len(wave_ref_resampled)}")
    print(f"len(wave_kukei_resampled) = {len(wave_kukei_resampled)}")

    # ---- t_offset_resampled 以降だけを「反射候補」として扱う ----
    a1 = t_offset_resampled
    wave_tail      = wave_kukei_resampled[a1:]   # 送信波の山は無視して、その後ろだけを見る
    wave_tail_ref  = wave_ref_resampled[a1:]     # 基準波形側も同様に反射部分だけを見る

    # ---- ピーク周辺の切り出し（反射部分の中でピーク探索：対象波形）----
    seg_kukei, peak_idx_local, seg_start_local, seg_end_local = kiritori2(
        wave_tail, left, right
    )

    # ---- 基準波形側も同じ条件でゲート ----
    seg_ref, _, _, _ = kiritori2(wave_tail_ref, left, right)

    # 再補間後の波形に対するグローバルインデックスに戻す（対象波形）
    peak_idx_global   = a1 + peak_idx_local
    seg_start_global  = a1 + seg_start_local
    seg_end_global    = a1 + seg_end_local

    print(f"ピーク位置 (resampled index) = {peak_idx_global}")
    print(f"切り出し区間 (resampled index): [{seg_start_global}, {seg_end_global})")
    print(f"len(seg_kukei) = {len(seg_kukei)}")
    print(f"len(seg_ref)   = {len(seg_ref)}")

    # ★ここでは DC オフセットは引かない
    seg_kukei_used = seg_kukei
    seg_ref_used   = seg_ref

    # ---- FFT 実行（対象波形 & 基準波形のスペクトル）----
    fft_kukei, freq = make_fftdata(seg_kukei_used, exp_dt)
    fft_ref,   _    = make_fftdata(seg_ref_used,   exp_dt)

    print(f"len(fft_kukei) = {len(fft_kukei)}")
    print(f"len(fft_ref)   = {len(fft_ref)}")
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

    # ================== プロット1：切り出し後の時間波形（対象） ==================

    plt.figure(figsize=(10, 4))
    t_seg = np.arange(len(seg_kukei_used)) * exp_dt
    plt.plot(t_seg * 1e6, seg_kukei_used)
    plt.xlabel("Time [µs]")
    plt.ylabel("Amplitude")
    plt.title("Gated waveform around reflected peak (kukei, resampled)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================== プロット2：0〜10 MHz の FFT スペクトル（基準＋対象を重ね描き） ==================

    band_full = (freq > 0) & (freq <= 10e6)

    plt.figure(figsize=(8, 6))
    plt.plot(freq[band_full] / 1e6, fft_ref[band_full],
             label="depth0 (smooth)")
    plt.plot(freq[band_full] / 1e6, fft_kukei[band_full],
             label=f"depth={f_depth*1e3:.2f} mm (defect)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude")
    plt.title("FFT spectrum (0–10 MHz, kukei, reflected part, resampled)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== 基準波形で割った 2〜8 MHz スペクトル ==================

    ratio_amp, freq_band = make_data(
        data0_resampled=wave_ref_resampled,
        data_resampled=wave_kukei_resampled,
        a1=a1,
    )
    freq_MHz = freq_band / 1e6

    plt.figure(figsize=(8, 6))
    plt.plot(freq_MHz, ratio_amp)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude ratio (depth / depth0)")
    plt.title("FFT ratio spectrum (2–8 MHz, kukei, reflected part, depth / depth0)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================== 参考：比スペクトルの 2–8 MHz 内ピーク周波数 ==================

    peak_idx_band = np.argmax(ratio_amp)
    peak_freq = freq_MHz[peak_idx_band]
    print(f"基準比スペクトルの 2-8 MHz 帯域内ピーク周波数 ≈ {peak_freq:.3f} MHz")

    print("矩形きずFFT（基準比）解析終了")
