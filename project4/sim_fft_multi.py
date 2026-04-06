##########################################################################
#  矩形きずシミュレーションCSVに対して
#    再補間 (exp_dt) → ピーク周り切り出し → ゼロパディングFFT
#    ＋ 基準波形（depth=0）で割ったスペクトルを算出する版
#    ＋ 平滑面(depth0)ときずあり(depth>0)のFFTを重ねて表示
#    ＋ 複数深さの振幅比を1枚のグラフに重ねて表示
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# ================== パス & 解析対象パラメータ（ここだけいじればOK） ==================

# シミュレーションCSVの場所
base_dir = r"C:\Users\cs16\Documents\Test_folder\tmp_output"   # 研究室PC用
# base_dir = r"C:\Users\hisay\OneDrive\ドキュメント\test_folder\tmp_output"   # 自宅PC用

# ファイル名プレフィックス（kukei_ori / kukei2 などを切り替えやすくする）
# file_prefix = "kukei1_ori_cupy"
# file_prefix = "kukei2_cupy"
file_prefix = "sankaku2_cupy"

# ピッチ [m]（FDTD側と揃える）
f_pitch = 1.25e-3

# デバッグ用：ひとつの深さについて確認する　[m]
f_depth_single = 0.10e-3   

# 複数深さの解析対象（ファイル名の depthXX の XX の部分）
depth_ids = [10 , 20]       

# ---- ここでファイル名をまとめて生成しておく ----
pitch_code = int(f_pitch * 1e5)              
depth_code_single = int(f_depth_single * 1e5) 

# 対象きず波形（単一深さ）
file_kukei_single = os.path.join(
    base_dir,
    f"{file_prefix}_pitch{pitch_code}_depth{depth_code_single}.csv"
)

# 基準波形（depth=0）
file_ref = os.path.join(
    base_dir,
    f"{file_prefix}_pitch{pitch_code}_depth0.csv"
)

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

fft_N = 2 ** 18           # FFT 長（固定）
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

    X = np.fft.fft(data_fft)
    freqs = np.linspace(0, 1.0 / dt_sample, n)
    magnitude = np.abs(X) / (n / 2)
    return magnitude, freqs


def extract_frequency_band(freq, fft_data, freq_min, freq_max):
    mask = (freq >= freq_min) & (freq <= freq_max)
    freq_band = freq[mask]
    fft_band  = fft_data[mask]
    return fft_band, freq_band


def interpolate_sim_one(data_raw):
    sim_t = np.arange(0, len(data_raw) * dt, dt)
    sim_t_new = np.arange(0, (len(data_raw) - 1) * dt, exp_dt)
    func = interp1d(sim_t, data_raw, kind='cubic')
    data_resampled = func(sim_t_new)
    return data_resampled


def make_data(data0_resampled, data_resampled, a1):
    """
    基準波形 data0_resampled と比較対象 data_resampled を使って、
    FFT 振幅の比 |FFT(data)/FFT(data0)|（2-8 MHz）を返す。
    """
    tail  = data_resampled[a1:]
    tail0 = data0_resampled[a1:]

    wave,  _, _, _ = kiritori2(tail,  left, right)
    wave0, _, _, _ = kiritori2(tail0, left, right)

    fft_data,  freq = make_fftdata(wave,  exp_dt)
    fft_data0, _    = make_fftdata(wave0, exp_dt)

    band,  freq_band = extract_frequency_band(freq, fft_data,  2e6, 8e6)
    band0, _         = extract_frequency_band(freq, fft_data0, 2e6, 8e6)

    ratio = np.abs(band / band0)
    return ratio, freq_band

# ================== メイン処理 ==================

if __name__ == "__main__":

    # ----------------------------------------------
    # まずは 1 つの深さについて、デバッグ用プロット
    # ----------------------------------------------

    print("読み込みファイル（矩形きずシミュレーション）:")
    print("  基準 :", file_ref)
    print("  対象 :", file_kukei_single)

    wave_ref   = np.loadtxt(file_ref,          delimiter=',', dtype=float)
    wave_kukei = np.loadtxt(file_kukei_single, delimiter=',', dtype=float)
    print(f"len(wave_ref)   = {len(wave_ref)}")
    print(f"len(wave_kukei) = {len(wave_kukei)}")

    wave_ref_resampled   = interpolate_sim_one(wave_ref)
    wave_kukei_resampled = interpolate_sim_one(wave_kukei)
    print(f"len(wave_ref_resampled)   = {len(wave_ref_resampled)}")
    print(f"len(wave_kukei_resampled) = {len(wave_kukei_resampled)}")

    a1 = t_offset_resampled
    wave_tail     = wave_kukei_resampled[a1:]
    wave_tail_ref = wave_ref_resampled[a1:]

    seg_kukei, peak_idx_local, seg_start_local, seg_end_local = kiritori2(
        wave_tail, left, right
    )
    seg_ref, _, _, _ = kiritori2(wave_tail_ref, left, right)

    peak_idx_global  = a1 + peak_idx_local
    seg_start_global = a1 + seg_start_local
    seg_end_global   = a1 + seg_end_local

    print(f"ピーク位置 (resampled index) = {peak_idx_global}")
    print(f"切り出し区間 (resampled index): [{seg_start_global}, {seg_end_global})")
    print(f"len(seg_kukei) = {len(seg_kukei)}")
    print(f"len(seg_ref)   = {len(seg_ref)}")

    seg_kukei_used = seg_kukei
    seg_ref_used   = seg_ref

    fft_kukei, freq = make_fftdata(seg_kukei_used, exp_dt)
    fft_ref,   _    = make_fftdata(seg_ref_used,   exp_dt)

    # ===== プロット0：再補間後の時間波形 + ゲート =====
    plt.figure(figsize=(10, 4))
    time_index = np.arange(len(wave_kukei_resampled))
    plt.plot(time_index, wave_kukei_resampled, label="resampled wave (kukei)")
    plt.axvline(a1, color="g", linestyle=":", label=f"t_offset_resampled={a1}")
    plt.axvline(peak_idx_global, color="r", linestyle="--", label="peak (in tail)")
    plt.axvspan(seg_start_global, seg_end_global, color="orange", alpha=0.3,
                label="gate region (reflection)")
    plt.xlabel("Resampled index")
    plt.ylabel("Amplitude")
    plt.title("Resampled waveform (kukei), t_offset and gated reflection region")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== プロット1：切り出し後の時間波形 =====
    plt.figure(figsize=(10, 4))
    t_seg = np.arange(len(seg_kukei_used)) * exp_dt
    plt.plot(t_seg * 1e6, seg_kukei_used)
    plt.xlabel("Time [µs]")
    plt.ylabel("Amplitude")
    plt.title("Gated waveform around reflected peak (kukei, resampled)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ===== プロット2：0〜10 MHz の FFT スペクトル（基準＋対象） =====
    band_full = (freq > 0) & (freq <= 10e6)
    plt.figure(figsize=(8, 6))
    plt.plot(freq[band_full] / 1e6, fft_ref[band_full],
             label="depth0 (smooth)")
    plt.plot(freq[band_full] / 1e6, fft_kukei[band_full],
             label=f"depth={f_depth_single*1e3:.2f} mm (defect)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude")
    plt.title("FFT spectrum (0-10 MHz, kukei, reflected part, resampled)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== 単一深さの振幅比 =====
    ratio_amp_single, freq_band_single = make_data(
        data0_resampled=wave_ref_resampled,
        data_resampled=wave_kukei_resampled,
        a1=a1,
    )
    freq_MHz_single = freq_band_single / 1e6

    plt.figure(figsize=(8, 6))
    plt.plot(freq_MHz_single, ratio_amp_single)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude ratio (depth / depth0)")
    plt.title("FFT ratio spectrum (2-8 MHz, single depth)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    peak_idx_band = np.argmax(ratio_amp_single)
    peak_freq = freq_MHz_single[peak_idx_band]
    print(f"[single] 2-8 MHz 帯域内ピーク周波数 ≈ {peak_freq:.3f} MHz")

    # ==========================================================
    # 複数深さの振幅比を1枚に重ねてプロット
    # ==========================================================

    ratios_all = []
    freq_band = None

    for d_id in depth_ids:
        file_kukei_d = os.path.join(
            base_dir,
            f"{file_prefix}_pitch{pitch_code}_depth{d_id}.csv"
        )
        print(f"[multi] 読み込み: {file_kukei_d}")

        wave_kukei_d = np.loadtxt(file_kukei_d, delimiter=',', dtype=float)
        wave_kukei_resampled_d = interpolate_sim_one(wave_kukei_d)

        ratio_d, freq_band_d = make_data(
            data0_resampled=wave_ref_resampled,
            data_resampled=wave_kukei_resampled_d,
            a1=a1,
        )

        ratios_all.append(ratio_d)
        if freq_band is None:
            freq_band = freq_band_d

    freq_MHz = freq_band / 1e6

    plt.figure(figsize=(8, 6))
    for d_id, ratio_d in zip(depth_ids, ratios_all):
        plt.plot(freq_MHz, ratio_d, label=f"depth ID {d_id}")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Normalized amplitude (depth / depth0)")
    plt.title("FFT ratio spectrum (2-8 MHz, multiple depths, simulation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("矩形きずFFT（基準比・複数深さ）解析終了")
