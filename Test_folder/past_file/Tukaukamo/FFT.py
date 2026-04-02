import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import pandas as pd

# ================== 各種パラメータ ==================

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

# FDTD の時間刻み（シミュレーション dt）
dt_sim = dx / cl / np.sqrt(6)
print(f"FDTD dt_sim = {dt_sim:.3e} [s] (約 {1.0/dt_sim/1e6:.3f} Msps)")

# ゲート幅（切り出しサイズ）
left  = 2 ** 10          # ピーク前に残す点数
right = (2 ** 10) * 3    # ピーク後に残す点数

# ゲート開始位置
exp_gate_start = 0      # 実験データの切り出し開始位置
sim_gate_start = 7560   # シミュレーションの切り出し開始位置

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
    data を fft_N 点にゼロパディング（前後に半分ずつ）して FFTを行う。
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


def load_experiment_average(base_dir, exp_files):
    """
    指定された実験 CSV (複数) を読み込んで平均化する。
    """
    waves = []
    time_exp = None

    for fname in exp_files:
        path = os.path.join(base_dir, fname)
        # print("読み込み (実験):", path) 

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

        # 最頻値を引く（ベースライン補正）
        mode_val = pd.Series(v).mode().iloc[0]
        v_detrended = v - mode_val

        waves.append(v_detrended)

    waves = np.vstack(waves)
    mean_wave = waves.mean(axis=0)

    return time_exp, mean_wave, waves


# ================== メイン処理 ==================

if __name__ == "__main__":

    # base_dir = r"C:\Users\cs16\Documents\Test_folder\tmp_output" #研究室PC
    base_dir = r"C:\Users\hisay\OneDrive\ドキュメント\Test_folder\tmp_output"   # 自宅PC

    # --- シミュレーション CSV ---
    # 必要な行のコメントを外して使用する

    # file_sim = os.path.join(base_dir, "kukei1_ori_cupy_pitch125_depth0.csv")
    # file_sim = os.path.join(base_dir, "kukei1_ori_cupy_pitch125_depth10.csv")
    file_sim = os.path.join(base_dir, "kukei1_ori_cupy_pitch125_depth20.csv")

    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch125_depth10.csv")
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch125_depth20.csv")
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch125_depth20_blocksize=1.csv")
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch125_depth20_step20.csv")
    
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch150_depth10.csv")
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch150_depth20.csv")
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch200_depth10.csv")
    # file_sim = os.path.join(base_dir, "sankaku2_cupy_pitch200_depth20.csv")
    
    # file_sim = os.path.join(base_dir, "kusabi_cupy_pitch125_depth10.csv")
    # file_sim = os.path.join(base_dir, "kusabi_cupy_pitch125_depth20.csv")
    # file_sim = os.path.join(base_dir, "kusabi_cupy_pitch150_depth10.csv")
    # file_sim = os.path.join(base_dir, "kusabi_cupy_pitch150_depth20.csv")
    # file_sim = os.path.join(base_dir, "kusabi_cupy_pitch200_depth10.csv")
    # file_sim = os.path.join(base_dir, "kusabi_cupy_pitch200_depth20.csv")
    
    
    # file_sim = os.path.join(base_dir, "hanen_pitch125_depth20.csv")


    # --- 実験 CSV（ファイル名をここに並べる） ---

    #############################################################

    '''
    # 平滑面の場合 
    exp_files = [
        "scope_106.csv", "scope_107.csv", "scope_108.csv", "scope_109.csv", "scope_110.csv",
        "scope_111.csv", "scope_112.csv", "scope_113.csv", "scope_114.csv", "scope_115.csv",
    ]

    '''

    #############################################################

    

    # kukeiきずの場合 (Pitch125_Depth20)
    exp_files = [
        "scope_5b1_50.csv", "scope_5b1_51.csv", "scope_5b1_52.csv", "scope_5b1_53.csv", "scope_5b1_54.csv", "scope_5b1_55.csv", 
        "scope_5b1_56.csv", "scope_5b1_57.csv", "scope_5b1_58.csv", "scope_5b1_59.csv", 
    ]

    

    #############################################################

    '''

    # sankakuきずの場合 (Pitch125_Depth10)
    exp_files = [
        "scope_52.csv", "scope_53.csv", "scope_54.csv", "scope_55.csv", "scope_56.csv",
        "scope_57.csv", "scope_58.csv", "scope_59.csv", "scope_60.csv", "scope_61.csv",
        "scope_62.csv", "scope_63.csv", "scope_64.csv", "scope_65.csv", "scope_66.csv",
        "scope_67.csv", "scope_68.csv", "scope_69.csv", "scope_70.csv", "scope_71.csv",
        "scope_72.csv", "scope_73.csv", "scope_74.csv", "scope_75.csv", "scope_76.csv",
    ]

    '''


    #############################################################
    
    '''
    
    # sankakuきずの場合 (Pitch125_Depth20)
    exp_files = [
        "scope_77.csv", "scope_78.csv", "scope_79.csv", "scope_80.csv", "scope_81.csv",
        "scope_96.csv", "scope_97.csv", "scope_98.csv", "scope_99.csv", "scope_100.csv",
        "scope_101.csv", "scope_102.csv", "scope_103.csv", "scope_104.csv", "scope_105.csv",
    ]

    '''

    #############################################################

    '''

    # sankakuきずの場合 (Pitch150_Depth10)
    exp_files = [
        "scope_116.csv", "scope_117.csv", "scope_118.csv", "scope_119.csv", "scope_120.csv",
        "scope_121.csv",
    ]

    '''

    #############################################################

    '''
    
    # sankakuきずの場合 (Pitch150_Depth20)
    exp_files = [
        "scope_122.csv", "scope_123.csv", "scope_124.csv", "scope_125.csv", "scope_126.csv",
        "scope_138.csv", "scope_139.csv", "scope_140.csv", "scope_141.csv", "scope_142.csv",
    ]

    '''


    #############################################################

    '''
    
    # sankakuきずの場合 (Pitch200_Depth10)
    exp_files = [
        "scope_127.csv", "scope_128.csv", "scope_129.csv", "scope_130.csv", "scope_131.csv",
        "scope_132.csv", "scope_143.csv", "scope_144.csv", "scope_145.csv", "scope_146.csv", 
        "scope_147.csv",
    ]

    '''
    

    #############################################################

    '''
    
    # sankakuきずの場合 (Pitch200_Depth20)
    exp_files = [
        "scope_133.csv", 
        "scope_134.csv", 
        "scope_135.csv", 
        "scope_136.csv", 
        "scope_137.csv",
        "scope_148.csv", 
        "scope_149.csv", 
        "scope_150.csv", 
        "scope_151.csv", 
        "scope_152.csv",
    ]

    '''

    #############################################################

    '''
    
    # kusabiきずの場合 (Pitch125_Depth10)
    exp_files = [
        "scope_153.csv", 
        "scope_154.csv", 
        "scope_155.csv",
        "scope_156.csv", 
        "scope_157.csv", 
        "scope_158.csv", 
        "scope_159.csv", 
        "scope_160.csv",
        "scope_161.csv", 
        "scope_162.csv",

    ]

    '''
    

    #############################################################

    '''
    
    # kusabiきずの場合 (Pitch125_Depth20)
    exp_files = [
        "scope_175.csv", 
        "scope_176.csv", 
        "scope_177.csv",
        "scope_178.csv", 
        "scope_179.csv", 
        "scope_180.csv", 
        "scope_181.csv", 
        "scope_182.csv",
    ]
    
    '''

    #############################################################

    '''
    
    # kusabiきずの場合 (Pitch150_Depth10)
    exp_files = [
        "scope_163.csv", 
        "scope_164.csv", 
        "scope_165.csv",
        "scope_166.csv", 
        "scope_167.csv", 
        "scope_168.csv", 
        "scope_169.csv", 
        "scope_170.csv",
        "scope_171.csv", 
        "scope_172.csv",

    ]

    '''

    #############################################################
    
    '''
    
    # kusabiきずの場合 (Pitch150_Depth20)
    exp_files = [
        "scope_183.csv", 
        "scope_184.csv", 
        "scope_185.csv",
        "scope_186.csv", 
        "scope_187.csv", 
        "scope_188.csv", 
        "scope_189.csv", 
        "scope_190.csv",
        "scope_191.csv", 
        "scope_192.csv",

    ]

    '''

    #############################################################
    
    '''
    
    # kusabiきずの場合 (Pitch200_Depth10)
    exp_files = [
        "scope_193.csv", 
        "scope_194.csv", 
        "scope_195.csv",
        "scope_196.csv", 
        "scope_197.csv", 
        "scope_198.csv", 
        "scope_199.csv", 
        "scope_200.csv",
        "scope_201.csv", 
        "scope_202.csv",

    ]

    '''

    #############################################################
    
    '''
    
    # kusabiきずの場合 (Pitch200_Depth20)
    exp_files = [
        "scope_203.csv", 
        "scope_204.csv", 
        "scope_205.csv",
        "scope_206.csv", 
        "scope_207.csv", 
        "scope_208.csv", 
        "scope_209.csv", 
        "scope_210.csv",
        "scope_211.csv", 

    ]

    '''

    #############################################################

    '''
    
    # hanenきずの場合 (Pitch125_Depth20)
    exp_files = [
        "scope_91.csv", "scope_92.csv", "scope_93.csv", "scope_94.csv", "scope_95.csv",
    ]

    '''
    

    #############################################################
    
    # === 基準データ（平滑面）の設定 ===
    
    # 平滑面シミュレーション
    file_sim_smooth = os.path.join(base_dir, "kukei1_ori_cupy_pitch125_depth0.csv")

    # 平滑面実験データ
    exp_files_smooth = [
        "scope_106.csv", "scope_107.csv", "scope_108.csv", "scope_109.csv", "scope_110.csv",
        "scope_111.csv", "scope_112.csv", "scope_113.csv", "scope_114.csv", "scope_115.csv",
    ]

    print("--- ターゲット(Defect) ---")
    print("シミュレーション CSV :", os.path.basename(file_sim))
    print(f"実験 CSV : {len(exp_files)} files")
    
    print("--- 基準(Smooth) ---")
    print("シミュレーション CSV :", os.path.basename(file_sim_smooth))
    print(f"実験 CSV : {len(exp_files_smooth)} files")

    # ---------- [Target] データ読み込み ----------
    wave_sim = np.loadtxt(file_sim, delimiter=",", dtype=float)
    # 実験波形（平均）
    time_exp, wave_exp_mean, waves_all = load_experiment_average(base_dir, exp_files)

    # ---------- [Smooth] データ読み込み（追加） ----------
    wave_sim_smooth = np.loadtxt(file_sim_smooth, delimiter=",", dtype=float)
    # 平滑面実験波形（平均）
    _, wave_exp_smooth_mean, _ = load_experiment_average(base_dir, exp_files_smooth)

    dt_exp = time_exp[1] - time_exp[0]
    print(f"推定 dt_exp = {dt_exp:.3e} [s]")

    # ================== シミュレーションの波形を実験の刻みに再補間 ==================

    # ターゲット
    wave_sim_resampled = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)

    # 基準（平滑面）
    wave_sim_smooth_resampled = interpolate_to_target_dt(wave_sim_smooth, dt_sim, dt_exp)

    print(f"len(wave_sim_resampled) = {len(wave_sim_resampled)}")

    # ================== 反射波部分のゲートの処理 ==================

    # --- [Target] 実験：exp_gate_start 以降 ---
    tail_exp = wave_exp_mean[exp_gate_start:]
    seg_exp, peak_idx_local_exp, seg_start_local_exp, seg_end_local_exp = kiritori2(
        tail_exp, left, right
    )
    peak_idx_global_exp  = exp_gate_start + peak_idx_local_exp
    seg_start_global_exp = exp_gate_start + seg_start_local_exp
    seg_end_global_exp   = exp_gate_start + seg_end_local_exp

    # --- [Target] シミュ：sim_gate_start 以降 ---
    tail_sim = wave_sim_resampled[sim_gate_start:]
    seg_sim, peak_idx_local_sim, seg_start_local_sim, seg_end_local_sim = kiritori2(
        tail_sim, left, right
    )
    peak_idx_global_sim  = sim_gate_start + peak_idx_local_sim
    seg_start_global_sim = sim_gate_start + seg_start_local_sim
    seg_end_global_sim   = sim_gate_start + seg_end_local_sim

    # --- [Smooth] 実験（同じゲート位置を使用） ---
    tail_exp_smooth = wave_exp_smooth_mean[exp_gate_start:]
    seg_exp_smooth, _, _, _ = kiritori2(tail_exp_smooth, left, right)

    # --- [Smooth] シミュ（同じゲート位置を使用） ---
    tail_sim_smooth = wave_sim_smooth_resampled[sim_gate_start:]
    seg_sim_smooth, _, _, _ = kiritori2(tail_sim_smooth, left, right)

    print(f"[EXP] peak index (global) = {peak_idx_global_exp}")
    print(f"[EXP] gate range (global) = [{seg_start_global_exp}, {seg_end_global_exp})")
    print(f"[SIM] peak index (global) = {peak_idx_global_sim}")
    print(f"[SIM] gate range (global) = [{seg_start_global_sim}, {seg_end_global_sim})")

    # ================== プロット0：実験平均波形＋ゲート ==================

    plt.figure(figsize=(10, 4))
    idx_exp = np.arange(len(wave_exp_mean))
    plt.plot(idx_exp, wave_exp_mean, label="experiment defect")
    plt.plot(idx_exp, wave_exp_smooth_mean, label="experiment smooth", alpha=0.5, linestyle="--")

    plt.axvline(exp_gate_start, color="g", linestyle=":", label=f"gate_search_start={exp_gate_start}")
    plt.axvline(peak_idx_global_exp, color="r", linestyle="--", label="exp peak")
    plt.axvspan(seg_start_global_exp, seg_end_global_exp, color="orange", alpha=0.3,
                label="exp gate")

    plt.xlabel("Index")
    plt.ylabel("Amplitude [V]")
    plt.title("Experimental waveform (averaged) and gated region")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== プロット1：シミュ波形＋ゲート ==================

    plt.figure(figsize=(10, 4))
    idx_sim = np.arange(len(wave_sim_resampled))
    plt.plot(idx_sim, wave_sim_resampled, label="simulation defect")
    plt.plot(idx_sim, wave_sim_smooth_resampled, label="simulation smooth", alpha=0.5, linestyle="--")

    plt.axvline(sim_gate_start, color="g", linestyle=":", label=f"gate_search_start={sim_gate_start}")
    plt.axvline(peak_idx_global_sim, color="r", linestyle="--", label="sim peak")
    plt.axvspan(seg_start_global_sim, seg_end_global_sim, color="orange", alpha=0.3,
                label="sim gate")

    plt.xlabel("Index")
    plt.ylabel("Amplitude [arb.]")
    plt.title("Simulation waveform (resampled) and gated region")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== FFT（どちらも dt_exp 基準） ==================

    # ターゲット (Defect)
    fft_exp, freq = make_fftdata(seg_exp, dt_exp)
    fft_sim, _    = make_fftdata(seg_sim, dt_exp)
    
    # 基準 (Smooth)
    fft_exp_smooth, _ = make_fftdata(seg_exp_smooth, dt_exp)
    fft_sim_smooth, _ = make_fftdata(seg_sim_smooth, dt_exp)

    band_full = (freq > 0) & (freq <= 10e6)

    # 生スペクトル
    plt.figure(figsize=(8, 5))
    plt.plot(freq[band_full] / 1e6, fft_exp[band_full], label="experiment")
    plt.plot(freq[band_full] / 1e6, fft_sim[band_full], label="simulation")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude")
    plt.title("FFT spectrum (0-10 MHz, gated)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== 2〜8 MHz 正規化スペクトル（Fig.6(a) 用） ==================

    band_2_8 = (freq >= 2e6) & (freq <= 8e6)
    freq_2_8_MHz = freq[band_2_8] / 1e6
    fft_exp_2_8 = fft_exp[band_2_8]
    fft_sim_2_8 = fft_sim[band_2_8]

    fft_exp_norm = fft_exp_2_8 / np.max(fft_exp_2_8)
    fft_sim_norm = fft_sim_2_8 / np.max(fft_sim_2_8)

    plt.figure(figsize=(8, 5))
    plt.plot(freq_2_8_MHz, fft_exp_norm, label="experiment (norm)")
    plt.plot(freq_2_8_MHz, fft_sim_norm, label="simulation (norm)")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Normalized amplitude")
    plt.title("Normalized FFT spectrum (2-8 MHz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== 傷あり / 平滑面 の比 ==================

    fft_exp_smooth_2_8 = fft_exp_smooth[band_2_8]
    fft_sim_smooth_2_8 = fft_sim_smooth[band_2_8]

    eps = 1e-12  # ゼロ割り防止用

    ratio_exp = fft_exp_2_8 / (fft_exp_smooth_2_8 + eps)
    ratio_sim = fft_sim_2_8 / (fft_sim_smooth_2_8 + eps)

    plt.figure(figsize=(8, 5))
    plt.plot(freq_2_8_MHz, ratio_exp, label="experiment: defect / smooth")
    plt.plot(freq_2_8_MHz, ratio_sim, label="simulation: defect / smooth")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude ratio (defect / smooth)")
    plt.title("Amplitude ratio spectrum (defect / smooth, 2-8 MHz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("シミュ vs 実験(平均) の FFT 比較終了 (Ratio Plot Added)")