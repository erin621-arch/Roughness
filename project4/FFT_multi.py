##########################################################################
#  シミュレーション CSV のみを使って
#    再補間 → ピーク周り切り出し → ゼロパディングFFT
#    ＋ 平滑面(depth=0) との振幅比を算出
#    ＋ 複数深さの振幅比を1枚のグラフに重ねて表示
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# ================== パス & 解析対象パラメータ（ここだけいじればOK） ==================

# ベースパス（FFT.py と同じ設定）
doc_path = r"C:/Users/cs16/Roughness/project4"  # 研究室PC
# doc_path = r"C:/Users/hisay/OneDrive/ドキュメント/Test_folder"   # 自宅PC

sim_base_dir = os.path.join(doc_path, "Simulation_Data")

# ============================================================
# ★ 1. 解析の周波数範囲設定
# ============================================================
freq_min = 2.0e6  # [Hz]
freq_max = 8.0e6  # [Hz]

# ============================================================
# ★ 2. 対象形状・ピッチの指定
# ============================================================
target_shape = "sankaku"  # "sankaku" / "kusabi" / "hanen"
target_pitch = 125         # ピッチ整数コード (1.25e-3 m → 125)

# ============================================================
# ★ 3. 深さの指定
# ============================================================
# 単一深さ確認用（デバッグ・波形目視用）
target_depth_single = 10   # 深さ整数コード (0.10e-3 m → 10)

# 複数深さ比較用（リストにすべて列挙）
target_depths = [10, 20]   # 深さ整数コードのリスト

# --- ファイル名自動生成 ---
target_sim_filename_single = f"{target_shape}_cupy_pitch{target_pitch}_depth{target_depth_single}.csv"
# step~など特殊ファイルの場合は手動で上書き:
# target_sim_filename_single = "sankaku_cupy_pitch125_depth10_step5.csv"

target_sim_filenames = [
    f"{target_shape}_cupy_pitch{target_pitch}_depth{d}.csv"
    for d in target_depths
]
# 特殊ファイルが混在する場合は手動で上書き:
# target_sim_filenames = [
#     "sankaku_cupy_pitch125_depth10.csv",
#     "sankaku_cupy_pitch125_depth20_step5.csv",
# ]

# ============================================================

# --- パス設定 ---
_shape_map = {"sankaku": "Sankaku", "kusabi": "Kusabi",
              "hanen": "Hanen", "smooth": "Smooth"}
sim_shape_dir = os.path.join(
    sim_base_dir, _shape_map.get(target_shape.lower(), target_shape.capitalize())
)

# 対象きず波形（単一深さ）
file_sim_single = os.path.join(sim_shape_dir, target_sim_filename_single)

# 平滑面シミュレーション（固定: Simulation_Data/Smooth/smooth_cupy_pitch125_depth0.csv）
smooth_sim_path = os.path.join(sim_base_dir, "Smooth", "smooth_cupy_pitch125_depth0.csv")

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

# ================== FFT 用パラメータ ==================

fft_N = 2 ** 18           # FFT 長
left  = 2 ** 10           # ピーク前に残す点数
right = (2 ** 10) * 3     # ピーク後に残す点数

# シミュレーション波形を再補間する目標サンプリング間隔（実験の dt に合わせた値）
dt_target = 0.0005e-6     # [s]

# 「このインデックス以降を反射波とみなす」開始位置（FFT.py の sim_gate_start に相当）
sim_gate_start = 8000

# ================== ユーティリティ関数 ==================

def kiritori2_safe(data_sample, left, right):
    """波形の最大値まわりに [left, right] だけ切り出し（範囲外は安全にクリップ）。"""
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend], datamaxhere, datamaxstart, datamaxend


def make_fftdata(data, dt_sample):
    """data を fft_N 点にゼロパディング（前後に半分ずつ）して FFT。"""
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
                [data_fft, np.zeros(n - len(data_fft))], axis=0
            )

    X = np.fft.fft(data_fft)
    freqs = np.linspace(0, 1.0 / dt_sample, n)
    magnitude = np.abs(X) / (n / 2)
    return magnitude, freqs


def interpolate_sim(data_raw):
    """シミュレーション波形を dt_target 刻みに再補間する。"""
    sim_t     = np.arange(0, len(data_raw) * dt, dt)
    sim_t_new = np.arange(0, (len(data_raw) - 1) * dt, dt_target)
    func = interp1d(sim_t, data_raw, kind='cubic')
    return func(sim_t_new)


def calc_ratio(wave_sim_smooth_rs, wave_sim_rs, gate_start):
    """
    平滑面波形 wave_sim_smooth_rs と対象波形 wave_sim_rs を使って
    FFT 振幅の比 |FFT(defect)| / |FFT(smooth)| を freq_min〜freq_max で返す。
    """
    tail       = wave_sim_rs[gate_start:]
    tail_smooth = wave_sim_smooth_rs[gate_start:]

    seg,        _, _, _ = kiritori2_safe(tail,        left, right)
    seg_smooth, _, _, _ = kiritori2_safe(tail_smooth, left, right)

    fft_data,   freq = make_fftdata(seg,        dt_target)
    fft_smooth, _    = make_fftdata(seg_smooth, dt_target)

    band = (freq >= freq_min) & (freq <= freq_max)
    ratio = np.abs(fft_data[band] / (fft_smooth[band] + 1e-12))
    return ratio, freq[band]


# ================== メイン処理 ==================

if __name__ == "__main__":

    print(f"【Target】 {target_shape} / {target_pitch}")
    print(f"  Sim Dir   : {sim_shape_dir}")
    print(f"  Smooth Sim: {smooth_sim_path}")
    print("-" * 60)

    # --- 平滑面シミュレーション読み込み ---
    if not os.path.exists(smooth_sim_path):
        raise FileNotFoundError(f"平滑面シミュレーションが見つかりません:\n  {smooth_sim_path}")
    wave_sim_smooth = np.loadtxt(smooth_sim_path, delimiter=",", dtype=float)
    wave_sim_smooth_rs = interpolate_sim(wave_sim_smooth)

    # -------------------------------------------------------
    # 単一深さ：デバッグ用プロット
    # -------------------------------------------------------
    print(f"[single] 読み込み: {file_sim_single}")
    wave_sim = np.loadtxt(file_sim_single, delimiter=",", dtype=float)
    wave_sim_rs = interpolate_sim(wave_sim)

    tail_sim    = wave_sim_rs[sim_gate_start:]
    tail_smooth = wave_sim_smooth_rs[sim_gate_start:]

    seg_sim,    peak_local, seg_start_local, seg_end_local = kiritori2_safe(tail_sim,    left, right)
    seg_smooth, _,          _,               _              = kiritori2_safe(tail_smooth, left, right)

    peak_global      = sim_gate_start + peak_local
    seg_start_global = sim_gate_start + seg_start_local
    seg_end_global   = sim_gate_start + seg_end_local

    print(f"  ピーク位置 (resampled index) = {peak_global}")
    print(f"  切り出し区間: [{seg_start_global}, {seg_end_global})")

    fft_sim,    freq = make_fftdata(seg_sim,    dt_target)
    fft_smooth, _    = make_fftdata(seg_smooth, dt_target)

    # -------------------------------------------------------
    # 複数深さ：振幅比を1枚に重ねてプロット
    # -------------------------------------------------------
    ratios_all  = []
    freq_band   = None
    labels_all  = []

    for fname in target_sim_filenames:
        fpath = os.path.join(sim_shape_dir, fname)
        print(f"[multi] 読み込み: {fpath}")

        wave_d    = np.loadtxt(fpath, delimiter=",", dtype=float)
        wave_d_rs = interpolate_sim(wave_d)

        ratio_d, freq_band_d = calc_ratio(wave_sim_smooth_rs, wave_d_rs, sim_gate_start)
        ratios_all.append(ratio_d)
        labels_all.append(fname)
        if freq_band is None:
            freq_band = freq_band_d

    plt.figure(figsize=(8, 5))
    for ratio_d, label in zip(ratios_all, labels_all):
        plt.plot(freq_band / 1e6, ratio_d, label=label)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude ratio (defect / smooth)")
    plt.title(f"FFT ratio spectrum ({freq_min/1e6:.1f}-{freq_max/1e6:.1f} MHz, multiple depths)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print("シミュレーション FFT（平滑面比・複数深さ）解析終了")
