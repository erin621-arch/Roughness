##########################################################################
#  source_new.py（コメント整理版）
#  - FDTD と同じ物性・dt の再計算
#  - FFT 用ユーティリティ（kiritori2, make_fftdata など）
#  - シミュレーション波形の再補間（interpolate_sim, interpolate_sim_one）
#  - 実験データ読み込み・加工（import_data など）
#  - シミュ/実験のスペクトル比較・RMSE評価
##########################################################################

import numpy as np
from scipy.interpolate import interp1d
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ============================================================
# 共有パラメータ・物性値
# ============================================================

fft_N = 2 ** 14

x_length = 0.02  # x方向の長さ m
y_length = 0.04  # y方向の長さ m
mesh_length = 1.0e-5  # m
nx = int(x_length / mesh_length)  # how many mesh
ny = int(y_length / mesh_length)

dx = x_length / nx  # mesh length m
dy = y_length / ny  # m

rho = 7840  # density kg/m^3
E = 206 * 1e9  # young percentage kg/ms^2
G = 80 * 1e9  # stiffness
V = 0.27  # poisson ratio

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P wave
ct = np.sqrt(G / rho)  # S wave
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)  # FDTD の時間刻み

# 実験側のサンプリング間隔（再補間先の dt）
exp_dt = 0.0005E-06

# ピーク周りのゲート幅
left = 2 ** 10
right = (2 ** 10) * 3

# 実験波形・シミュレーション波形の切り出し用インデックス
# （exp_a*, sim_a* は別用途の境界インデックス）
exp_a1 = 0
exp_a2 = 3730
exp_a3 = 12000

sim_a1 = 6000
sim_a2 = 15000
sim_a3 = sim_a2 + exp_a3 - exp_a2

data_size = exp_a3 - exp_a2

# ============================================================
# ピッチ・深さ・粗さパラメータ（実験条件セット）
# ============================================================

depth_125 = [2, 5, 10, 14, 18, 20, 23]
depth_150 = [3, 4, 6, 10, 15, 20]
depth_175 = [3, 4, 6, 10, 15, 21]
depth_200 = [2, 4, 6, 10, 17, 20, 22, 28]

pitch_125 = [10, 30, 70, 10, 20, 80, 30]
pitch_150 = [150, 160, 170, 70, 80, 90]
pitch_175 = [180, 190, 200, 100, 110, 120]
pitch_200 = [40, 50, 60, 90, 40, 100, 50, 60]

pitch_list = [1250, 1500, 1750, 2000]

rough_list = {
    1250: len(pitch_125),
    1500: len(pitch_150),
    1750: len(pitch_175),
    2000: len(pitch_200)
}

depth_index = {
    0: 0,
    1250: len(depth_125),
    1500: len(depth_150),
    1750: len(depth_175),
    2000: len(depth_200)
}

rough_list_all = {
    1250: pitch_125,
    1500: pitch_150,
    1750: pitch_175,
    2000: pitch_200
}

depth_index_all = {
    1250: depth_125,
    1500: depth_150,
    1750: depth_175,
    2000: depth_200
}

# ============================================================
# FFT 用ユーティリティ
# ============================================================

def kiritori2(data_sumple, left, right):
    """波形の最大値周り [left, right] を切り出す（ピークゲート）"""
    datamaxhere = np.nanargmax(data_sumple)
    datamaxstart = datamaxhere - left
    datamaxend = datamaxhere + right
    return data_sumple[datamaxstart:datamaxend]


def make_fftdata(data, dt_sumple):
    """
    シンプルな FFT:
    - data を fft_N 点にゼロパディング（前後に半分ずつ）
    - np.fft.fft で複素 FFT
    - 振幅スペクトルと周波数軸を返す
    """
    howmanyzero = fft_N - len(data)
    data_fft = np.concatenate(
        [np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))],
        0
    )
    yf = np.fft.fft(data_fft) / (fft_N / 2)
    freq = np.linspace(0, 1.0 / dt_sumple, fft_N)
    return np.abs(yf), freq

# ============================================================
# シミュレーション波形の読み込み・再補間（tri_fft_kukei interpolate 系に対応）
# ============================================================

def simu_data_import(filenames):
    """
    シミュレーション CSV を複数読み込んで 2D 配列にまとめる。
    先頭に pitch100_depth0（基準波形）を必ず入れる。
    """
    data0 = np.loadtxt(
        f"C:\\Users\\Fujii Kotaro\\project1\\data_all\\cupy_pitch100_depth0.csv"
    )
    data_set_raw = data0
    for filename in filenames:
        data = np.loadtxt(
            f"C:\\Users\\Fujii Kotaro\\project1\\data_all\\{filename}"
        )
        print(filename)
        data_set_raw = np.vstack([data_set_raw, data])
    return data_set_raw


def interpolate_sim(data_set_raw):
    """
    シミュレーション波形（複数本）を dt → exp_dt に再補間。
    tri_fft_kukei の interpolate_sim_one の「バッチ版」。
    """
    sim_t = np.arange(0, len(data_set_raw[0]) * dt, dt)
    sim_t_new = np.arange(0, (len(data_set_raw[0]) - 1) * dt, exp_dt)
    data_set_raw_new = np.zeros((data_set_raw.shape[0], len(sim_t_new)))
    for i in range(data_set_raw.shape[0]):
        func = interp1d(sim_t, data_set_raw[i], kind='cubic')
        data_set_raw_new[i] = func(sim_t_new)
    return data_set_raw_new


def interpolate_sim_one(data_set_raw):
    """
    シミュレーション波形（1本）を dt → exp_dt に再補間。
    tri_fft_kukei の interpolate_sim_one と対応。
    """
    sim_t = np.arange(0, len(data_set_raw) * dt, dt)
    sim_t_new = np.arange(0, (len(data_set_raw) - 1) * dt, exp_dt)
    data_set_raw_new = np.zeros((data_set_raw.shape[0], len(sim_t_new)))
    func = interp1d(sim_t, data_set_raw, kind='cubic')
    data_set_raw_new = func(sim_t_new)
    return data_set_raw_new


def interpolate_sim_one_inverse(data_set_raw):
    """
    exp_dt サンプリング → dt サンプリングに戻す逆再補間。
    （NaN を除いた部分のみを使う）
    """
    # nanではない最初のインデックスを見つける
    first_valid = np.where(~np.isnan(data_set_raw))[0][0]
    # nanではない部分だけを使用
    valid_data = data_set_raw[first_valid:]
    sim_t = np.arange(first_valid * exp_dt, len(data_set_raw) * exp_dt, exp_dt)
    sim_t_new = np.arange(first_valid * exp_dt, (len(data_set_raw) - 1) * exp_dt, dt)
    func = interp1d(sim_t, valid_data, kind='cubic')
    data_set_raw_new = func(sim_t_new)
    return data_set_raw_new

# ============================================================
# FFT スペクトルの周波数帯抽出（1D / tri_fft_kukei の extract_frequency_band と対応）
# ============================================================

def extract_frequency_band(freq, fft_data, freq_min, freq_max):
    """
    特定の周波数帯のデータを抽出する関数
    （1本のスペクトル用：tri_fft_kukei のバンド抽出と同じイメージ）
    """
    mask = (freq >= freq_min) & (freq <= freq_max)
    freq_band = freq[mask]
    fft_band = fft_data[mask]
    return fft_band, freq_band

# ============================================================
# シミュレーション FFT データセット作成（基準波形で正規化）
# ============================================================

def make_data(data0, data, a1):
    """
    data0: 基準波形（depth=0 など）
    data : 比較対象波形
    a1   : 解析開始インデックス（t_offset 相当）
    - 両方とも a1 以降を使い、
    - ピーク周りを kiritori2 で切り出し、
    - ゼロパディング FFT → 2〜8 MHz 抽出 → data/data0 で正規化
    """
    wave = kiritori2(data[a1:], left, right)
    wave0 = kiritori2(data0[a1:], left, right)
    fft_data, freq = make_fftdata(wave, exp_dt)
    fft_data0, freq = make_fftdata(wave0, exp_dt)
    return (
        extract_frequency_band(freq, fft_data, 2e06, 8e06)[0]
        / extract_frequency_band(freq, fft_data0, 2e06, 8e06)[0],
        extract_frequency_band(freq, fft_data, 2e06, 8e06)[1],
    )


def make_dataset_sim(filenames, a1):
    """
    シミュレーション（複数ファイル）の FFT データセット生成。
    - simu_data_import で基準 + 各ケース読込
    - interpolate_sim で再補間
    - make_data で基準波形に対する正規化スペクトルを作成
    """
    data_set_raw = interpolate_sim(simu_data_import(filenames))
    data_test, freq = make_data(data_set_raw[0], data_set_raw[0], a1)
    # simu_data_import の先頭が基準なので -1
    data_set = np.zeros((data_set_raw.shape[0] - 1, len(data_test)))
    for i in range(data_set_raw.shape[0] - 1):
        print(i)
        data_set[i], freq = make_data(data_set_raw[0], data_set_raw[i + 1], a1)
    return data_set, freq

# ============================================================
# 実験データの読み込み・整形（import_data）
# ============================================================

def extract_numbers(filename):
    """
    ファイル名から pitch, depth の数値部分を抽出する関数。
    例: '...pitch125_depth10...' → (125, 10)
    """
    pitch_match = re.search(r'pitch(\d+)', filename)
    depth_match = re.search(r'depth(\d+)', filename)
    pitch = int(pitch_match.group(1)) if pitch_match else None
    depth = int(depth_match.group(1)) if depth_match else None
    return pitch, depth


def import_data(pitch, depth):
    """
    実験 CSV（指定 pitch, depth）をすべて読み込んで
    - 冒頭2行を削除
    - 2列目（振幅）からモード（代表値）を引いて DC オフセット除去
    - 全ショット平均波形 data_m を返す
    """
    data_tmp0 = []
    data_tmp1 = []
    data_tmp2 = []
    data_tmp3 = []
    data_tmp4 = []
    number_max = []
    data = []

    csv_files = glob.glob(
        os.path.join(
            f"C:\\Users\\Fujii Kotaro\\project2\\experience_new\\{pitch}_{depth}",
            "*.csv"
        )
    )

    i = 0
    for csv_file in csv_files:
        print(f"importing : {csv_file}")
        data_tmp0.append(
            pd.read_csv(csv_file, encoding='SHIFT-JIS', header=None)
        )
        # 先頭2行を削除（時間軸などのヘッダ）
        data_tmp1.append(data_tmp0[i].drop(data_tmp0[i].index[[0, 1]]))
        data_tmp1[i] = data_tmp1[i].astype(float)
        data_tmp2.append(data_tmp1[i].iloc[:, 0])  # time
        data_tmp3.append(data_tmp1[i].iloc[:, 1])  # value
        number_max.append(data_tmp2[i].count())
        # モード値で DC オフセット除去
        data_tmp4.append([data_tmp3[i].mode().iloc[0]] * number_max[i])
        data.append(data_tmp3[i] - data_tmp4[i])
        i += 1

    data_arr = np.array(data)
    data_m = data_arr.mean(0)  # 全ショット平均
    return data_m, data_arr

# ============================================================
# ピッチ・粗さ → 深さの対応（ヘルパー）
# ============================================================

def depthrough(pitch, rough):
    match pitch:
        case 0:
            return 0, 0
        case 1250:
            return depth_125[rough]
        case 2000:
            return depth_200[rough]
        case 1500:
            return depth_150[rough]
        case 1750:
            return depth_175[rough]
        case _:
            return 0

def make_dataset_exp_y_new(pitch, rough):
    """
    実験側の「入力ベクトル」（ピッチ・深さ）を作る。
    pitch : μm 単位 → /10 して 0.1mm 単位にしている。
    """
    depth_set = np.full(1, float(depthrough(pitch, rough)))
    pitch_set = np.full(1, float(pitch))
    return np.column_stack((pitch_set / 10, depth_set))

# ============================================================
# 実験 FFT データセット作成
# ============================================================

def make_dataset_exp_new(pitch, depth, a1):
    """
    実験側の FFT データセット 1本分を作る。
    - (0,0) を基準（平板）として mean0
    - 指定 (pitch, depth) の mean を対象として、
      make_data(mean0, mean, a1) で基準比スペクトルを作成。
    """
    mean, data_set_raw_exp = import_data(pitch, depth)
    mean0, data_set_raw_exp0 = import_data(0, 0)
    data_test_exp, freq = make_data(mean0, data_set_raw_exp[0], a1)
    data_set_exp = np.zeros((1, len(data_test_exp)))
    data_set_exp[0], freq = make_data(mean0, mean, a1)
    return data_set_exp

# ============================================================
# 可視化：シミュレーション側スペクトル（Normalized amplitude vs f）
# （tri_fft_kukei の「1本だけ版」を多本数＋plot したもの）
# ============================================================

def select_pitch_and_image(pitch, depth_list):
    """
    シミュレーションの FFT を depth_list について描画する。
    - 基準: pitch125_depth0（シミュレーション）
    - interpolate_sim_one → [8000:] → kiritori2 → FFT
    - yf / yf_ref を 2〜8 MHz の範囲で比較（プロットは Hz ベース）
    """
    sim_data_ref = np.loadtxt(
        r"C:\Users\Fujii Kotaro\project2\data_all\cupy_pitch125_depth0.csv"
    )
    yf_ref, freq_ref = make_fftdata(
        kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right),
        exp_dt
    )

    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))

    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(
            f"C:\\Users\\Fujii Kotaro\\project2\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv"
        )
        yf[i], freq[i] = make_fftdata(
            kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right),
            exp_dt
        )

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{int(depth_list[i]) / 100} mm")
    plt.xlim(2000000, 8000000)
    plt.ylim(0, 1.7)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Frequency Hz")
    plt.legend(ncol=2, loc='upper left')
    plt.tight_layout()
    plt.show()

# ============================================================
# 可視化：実験側スペクトル（Normalized amplitude vs f）
# ============================================================

def select_pitch_and_image_exp(pitch, depth_list):
    """
    実験の FFT を depth_list について描画する。
    - 基準: (pitch=0, depth=0) の平板実験
    - kiritori2 → FFT → yf / yf_ref
    """
    sim_data_ref, gomi = import_data(0, 0)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)

    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))

    for i in range(len(depth_list)):
        sim_data[i], gomi = import_data(pitch, depth_list[i])
        yf[i], freq[i] = make_fftdata(
            kiritori2(sim_data[i], left, right),
            exp_dt
        )

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{depth_list[i] / 100} mm")
    plt.xlim(2000000, 8000000)
    plt.ylim(0, 1.7)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Frequency Hz")
    plt.legend(ncol=2, loc='upper left')
    plt.tight.tight_layout()
    plt.show()

# ============================================================
# シミュ/実験のスペクトル配列を返すだけの関数（数値評価用）
# ============================================================

def select_sim(pitch, depth_list):
    """
    シミュレーション側の (freq, yf, yf_ref) を返す。
    プロットは別でやることを想定。
    """
    sim_data_ref = np.loadtxt(
        r"C:\Users\Fujii Kotaro\project2\data_all\cupy_pitch125_depth0.csv"
    )
    yf_ref, freq_ref = make_fftdata(
        kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right),
        exp_dt
    )

    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))

    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(
            f"C:\\Users\\Fujii Kotaro\\project2\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv"
        )
        yf[i], freq[i] = make_fftdata(
            kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right),
            exp_dt
        )

    return freq, yf, yf_ref


def select_exp(pitch, depth_list):
    """
    実験側の (freq, yf, yf_ref) を返す。
    """
    sim_data_ref, gomi = import_data(0, 0)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)

    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))

    for i in range(len(depth_list)):
        sim_data[i], gomi = import_data(pitch, depth_list[i])
        yf[i], freq[i] = make_fftdata(
            kiritori2(sim_data[i], left, right),
            exp_dt
        )

    return freq, yf, yf_ref

# ============================================================
# 2次元スペクトルの周波数帯抽出・誤差評価（RMSE・列統計）
# ============================================================

def extract_frequency_band_2d(freq, fft_data_2d, freq_min, freq_max):
    """
    2次元スペクトルデータから特定の周波数帯のデータを抽出する関数
    - 行: ケース
    - 列: 周波数
    """
    mask = (freq >= freq_min) & (freq <= freq_max)
    freq_band = freq[mask]
    fft_band_2d = fft_data_2d[:, mask]
    return fft_band_2d, freq_band


def calculate_rmse_with_frequency_band(freq, array1, array2, freq_min, freq_max):
    """
    指定された周波数帯域内で、2つの 2D 配列の「各行ごと RMSE」を計算する。
    - array1, array2: shape (N_case, N_freq)
    """
    if array1.shape != array2.shape:
        raise ValueError("両方の配列は同じ形状である必要があります")

    masked_array1, freq_band = extract_frequency_band_2d(freq, array1, freq_min, freq_max)
    masked_array2, _ = extract_frequency_band_2d(freq, array2, freq_min, freq_max)

    squared_diff = np.square(masked_array1 - masked_array2)
    mean_squared_diff = np.mean(squared_diff, axis=1)
    row_rmse = np.sqrt(mean_squared_diff)

    return row_rmse, masked_array1, masked_array2, freq_band


def calculate_column_statistics_with_frequency_band(freq, array1, array2, freq_min, freq_max):
    """
    指定周波数帯に制限したあと、列（周波数ごと）の誤差統計を計算する。
    - RMSE
    - max/min/mean/median/std
    """
    if array1.shape != array2.shape:
        raise ValueError("両方の配列は同じ形状である必要があります")

    masked_array1, freq_band = extract_frequency_band_2d(freq, array1, freq_min, freq_max)
    masked_array2, _ = extract_frequency_band_2d(freq, array2, freq_min, freq_max)

    squared_diff = np.square(masked_array1 - masked_array2)
    mean_squared_diff = np.mean(squared_diff, axis=0)
    column_rmse = np.sqrt(mean_squared_diff)

    abs_diff = np.abs(masked_array1 - masked_array2)

    stats = {
        'rmse': column_rmse,
        'max_error': np.max(abs_diff, axis=0),
        'min_error': np.min(abs_diff, axis=0),
        'mean_error': np.mean(abs_diff, axis=0),
        'median_error': np.median(abs_diff, axis=0),
        'std_error': np.std(abs_diff, axis=0),
    }

    max_rmse_index = np.argmax(column_rmse)
    stats['max_rmse_column_index'] = max_rmse_index
    stats['max_rmse_frequency'] = freq_band[max_rmse_index]

    return stats, masked_array1, masked_array2, freq_band

# ============================================================
# 波数軸版 FFT / プロット（make_fftdata_wavy, select_*_wavy）
# ============================================================

def make_fftdata_wavy(data, dt_sample, wave_speed=cl, wavenumber_type='standard'):
    """
    FFT を計算し、周波数 → 波数に変換して返す。
    wavenumber_type:
      - 'standard': k = f / c [1/m]
      - 'angular' : k = 2π f / c [rad/m]
    """
    howmanyzero = fft_N - len(data)
    data_fft = np.concatenate(
        [np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))],
        0
    )
    yf = np.fft.fft(data_fft) / (fft_N / 2)
    freq = np.linspace(0, 1.0 / dt_sample, fft_N)

    if wavenumber_type == 'angular':
        k_values = 2 * np.pi * freq / wave_speed
    else:
        k_values = freq / wave_speed

    return np.abs(yf), k_values


def select_pitch_and_image_wavy(pitch, depth_list):
    """
    シミュレーションの FFT を「波数軸」で描画するバージョン。
    （周波数 → k = f / c に変換）
    """
    sim_data_ref = np.loadtxt(
        r"C:\Users\Fujii Kotaro\project2\data_all\cupy_pitch125_depth0.csv"
    )
    yf_ref, freq_ref = make_fftdata_wavy(
        kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right),
        exp_dt
    )

    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))

    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(
            f"C:\\Users\\Fujii Kotaro\\project2\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv"
        )
        # ここは make_fftdata になっているが、波数軸にしたければ make_fftdata_wavy にしてもよい
        yf[i], freq[i] = make_fftdata(
            kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right),
            exp_dt
        )

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{int(depth_list[i]) / 100} mm")
    plt.xlim(2000000 / cl, 8000000 / cl)
    plt.ylim(0, 1.5)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Wave number m^-1")
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_pitch_and_image_exp_wavy(pitch, depth_list):
    """
    実験の FFT を「波数軸」で描画するバージョン。
    """
    sim_data_ref, gomi = import_data(0, 0)
    yf_ref, freq_ref = make_fftdata_wavy(
        kiritori2(sim_data_ref, left, right),
        exp_dt
    )

    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))

    for i in range(len(depth_list)):
        sim_data[i], gomi = import_data(pitch, depth_list[i])
        yf[i], freq[i] = make_fftdata(
            kiritori2(sim_data[i], left, right),
            exp_dt
        )

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{depth_list[i] / 100} mm")
    plt.xlim(2000000 / cl, 8000000 / cl)
    plt.ylim(0, 1.5)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Wave number m^-1")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# メイン波形 vs 前半・後半での FFT を比較する補助関数
# ============================================================

def make_fftdata_main_tale(data, dt, boader):
    """
    1本の時間波形 data について:
    - 全体
    - [0:boader]
    - [boader:]
    の 3パターンで FFT を計算して返す。
    """
    yf, freq = np.zeros((3, fft_N)), np.zeros((3, fft_N))

    howmanyzero0 = fft_N - len(data)
    data_fft0 = np.concatenate(
        [np.zeros(int(howmanyzero0 / 2)), data, np.zeros(int(howmanyzero0 / 2))],
        0
    )
    yf0 = np.fft.fft(data_fft0) / (fft_N / 2)
    freq1 = np.linspace(0, 1.0 / dt, fft_N)

    howmanyzero1 = fft_N - len(data[:boader])
    data_fft1 = np.concatenate(
        [np.zeros(int(howmanyzero1 / 2)), data[:boader], np.zeros(int(howmanyzero1 / 2))],
        0
    )
    yf1 = np.fft.fft(data_fft1) / (fft_N / 2)

    howmanyzero2 = fft_N - len(data[boader:])
    data_fft2 = np.concatenate(
        [np.zeros(int(howmanyzero2 / 2)), data[boader:], np.zeros(int(howmanyzero2 / 2))],
        0
    )
    yf2 = np.fft.fft(data_fft2) / (fft_N / 2)

    yf[0], freq[0] = np.abs(yf0), freq1
    yf[1], freq[1] = np.abs(yf1), freq1
    yf[2], freq[2] = np.abs(yf2), freq1

    return yf, freq
