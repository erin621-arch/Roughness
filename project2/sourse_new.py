import numpy as np
from scipy.interpolate import interp1d
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

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

dt = dx / cl / np.sqrt(6)  # time mesh
exp_dt = 0.0005E-06
left = int(2 ** 10)
right = int((2 ** 10) * 3)
exp_a1 = 0
exp_a2 = 3730
exp_a3 = 12000

sim_a1 = 6000
sim_a2 = 15000
sim_a3 = sim_a2 + exp_a3 - exp_a2

data_size = exp_a3 - exp_a2


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


def kiritori2(data_sumple, left, right):
    L_target = left + right # 目標の長さ
    datamaxhere = np.nanargmax(data_sumple)
    datamaxstart = datamaxhere - left
    datamaxend = datamaxhere + right
    cut_data = data_sumple[datamaxstart:datamaxend]
    
    if len(cut_data) != L_target:
        # 長さが異なる場合、ゼロパディングで強制的に目標長に合わせる
        padding_needed = L_target - len(cut_data)
        if padding_needed > 0:
            return np.pad(cut_data, (0, padding_needed), mode='constant')
        else: # 長すぎる場合は切り捨てる
            return cut_data[:L_target]
    print(f"cut_data.shape: {cut_data.shape}")
    return cut_data


def make_fftdata(data, dt_sumple):
    howmanyzero = fft_N - len(data)
    data_fft = np.concatenate([np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))], 0)
    yf = np.fft.rfft(data_fft) / (fft_N / 2)
    freq = np.fft.rfftfreq(fft_N, d=dt_sumple)
    return np.abs(yf), freq

def simu_data_import(filenames):
    data0 = np.loadtxt(
        f"C:\\Users\\Fujii Kotaro\\project1\\data_all\\cupy_pitch100_depth0.csv")
    data_set_raw = data0
    for filename in filenames:
        data = np.loadtxt(
            f"C:\\Users\\Fujii Kotaro\\project1\\data_all\\{filename}")
        print(filename)
        data_set_raw = np.vstack([data_set_raw, data])
    return data_set_raw


def interpolate_sim(data_set_raw):
    sim_t = np.arange(0, len(data_set_raw[0]) * dt, dt)
    sim_t_new = np.arange(0, (len(data_set_raw[0]) - 1) * dt, exp_dt)
    data_set_raw_new = np.zeros((data_set_raw.shape[0], len(sim_t_new)))
    for i in range(data_set_raw.shape[0]):
        func = interp1d(sim_t, data_set_raw[i], kind='cubic')
        data_set_raw_new[i] = func(sim_t_new)
    return data_set_raw_new

def interpolate_sim_one(data_set_raw):
    sim_t = np.arange(0, len(data_set_raw) * dt, dt)
    sim_t_new = np.arange(0, (len(data_set_raw) - 1) * dt, exp_dt)
    data_set_raw_new = np.zeros((data_set_raw.shape[0], len(sim_t_new)))
    
    func = interp1d(sim_t, data_set_raw, kind='cubic')
    data_set_raw_new = func(sim_t_new)
    return data_set_raw_new


def interpolate_sim_one_inverse(data_set_raw):
    # nanではない最初のインデックスを見つける
    first_valid = np.where(~np.isnan(data_set_raw))[0][0]
    
    # nanではない部分だけを使用
    valid_data = data_set_raw[first_valid:]
    sim_t = np.arange(first_valid * exp_dt, len(data_set_raw) * exp_dt, exp_dt)
    sim_t_new = np.arange(first_valid * exp_dt, (len(data_set_raw) - 1) * exp_dt, dt)
    
    func = interp1d(sim_t, valid_data, kind='cubic')
    data_set_raw_new = func(sim_t_new)
    return data_set_raw_new

def extract_frequency_band(freq, fft_data, freq_min, freq_max):
    """
    特定の周波数帯のデータを抽出する関数
    
    Parameters:
    -----------
    freq : array-like
        周波数配列
    fft_data : array-like
        振幅スペクトル配列（freqと同じ長さ）
    freq_min : float
        抽出したい周波数の下限
    freq_max : float
        抽出したい周波数の上限
    
    Returns:
    --------
    freq_band : ndarray
        抽出された周波数配列
    fft_band : ndarray
        抽出された振幅スペクトル配列
    """
    # 指定された周波数範囲のインデックスを取得
    mask = (freq >= freq_min) & (freq <= freq_max)
    
    # マスクを使って対応するデータを抽出
    freq_band = freq[mask]
    fft_band = fft_data[mask]
    
    return fft_band, freq_band

def make_data(data0, data, a1):
    wave = kiritori2(data[a1:], left, right)
    wave0 = kiritori2(data0[a1:], left, right)
    fft_data, freq = make_fftdata(wave, exp_dt)
    fft_data0, freq = make_fftdata(wave0, exp_dt)
    return extract_frequency_band(freq, fft_data, 2e06, 8e06)[0]/extract_frequency_band(freq, fft_data0, 2e06, 8e06)[0], extract_frequency_band(freq, fft_data, 2e06, 8e06)[1]


def make_dataset_sim(filenames, a1): #  ここ注意
    data_set_raw = interpolate_sim(simu_data_import(filenames))
    data_test, freq = make_data(data_set_raw[0], data_set_raw[0], a1)
    data_set = np.zeros((data_set_raw.shape[0]-1, len(data_test))) #  simu_data_importでやっちゃってる
    for i in range(data_set_raw.shape[0] - 1):
        print(i)
        data_set[i], freq = make_data(data_set_raw[0], data_set_raw[i+1], a1)
    return data_set, freq


# 数字を抽出する関数
def extract_numbers(filename):
    pitch_match = re.search(r'pitch(\d+)', filename)
    depth_match = re.search(r'depth(\d+)', filename)
    
    pitch = int(pitch_match.group(1)) if pitch_match else None
    depth = int(depth_match.group(1)) if depth_match else None
    
    return pitch, depth


def import_data(pitch, depth):
    data_tmp0 = []
    data_tmp1 = []
    data_tmp2 = []
    data_tmp3 = []
    data_tmp4 = []
    number_max = []
    data = []

    csv_files = glob.glob(os.path.join(f"C:\\Users\\manat\\project2\\experience_new\\{pitch}_{depth}", "*.csv"))
    i = 0
    for csv_file in csv_files:
        print(f"importing : {csv_file}")
        data_tmp0.append(
            pd.read_csv(csv_file, encoding='SHIFT-JIS', header=None))
        
        data_tmp1.append(data_tmp0[i].drop(data_tmp0[i].index[[0, 1]]))
        data_tmp1[i] = data_tmp1[i].astype(float)
        data_tmp2.append(data_tmp1[i].iloc[:, 0])
        data_tmp3.append(data_tmp1[i].iloc[:, 1])
        number_max.append(data_tmp2[i].count())
        data_tmp4.append([data_tmp3[i].mode().iloc[0]] * number_max[i])
        data.append(data_tmp3[i] - data_tmp4[i])
        i+=1
    data_arr = np.array(data)
    data_m = data_arr.mean(0)
    return data_m, data_arr


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
    depth_set = np.full(1, float(depthrough(pitch, rough)))
    pitch_set = np.full(1, float(pitch))
    return np.column_stack((pitch_set/10, depth_set))


def make_dataset_exp_new(pitch, depth, a1):
    mean, data_set_raw_exp = import_data(pitch, depth)
    mean0, data_set_raw_exp0 = import_data(0, 0)
    data_test_exp, freq = make_data(mean0, data_set_raw_exp[0], a1)
    data_set_exp = np.zeros((1, len(data_test_exp)))
    data_set_exp[0], freq = make_data(mean0, mean, a1)
    return data_set_exp


def select_pitch_and_image(pitch, depth_list):
    sim_data_ref = np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth0.csv")
    yf_ref, freq_ref = make_fftdata(kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(f"C:\\Users\\manat\\project2\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv")
        yf[i], freq[i] = make_fftdata(kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right), exp_dt)

    plt.rcParams["font.size"] = 16
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{int(depth_list[i])/100} mm")
    plt.xlim(2000000, 8000000)
    plt.ylim(0, 1.7)
    plt.ylabel("Normalized amplitude", fontsize=17)
    plt.xlabel("Frequency [Hz]", fontsize=17)
    plt.legend(ncol=2, loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.savefig(fr"C:\Users\manat\project2\tmp_output\sim_yfs_{pitch}.eps", format="eps", dpi=300)
    plt.show()



def select_pitch_and_image_exp(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 0)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i], gomi = import_data(pitch, depth_list[i])
        yf[i], freq[i] = make_fftdata(kiritori2(sim_data[i], left, right), exp_dt)

    plt.rcParams["font.size"] = 16
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{depth_list[i]/100} mm")
    plt.xlim(2000000, 8000000)
    plt.ylim(0, 1.7)
    plt.ylabel("Normalized amplitude", fontsize=17)
    plt.xlabel("Frequency [Hz]", fontsize=17)
    plt.legend(ncol=2, loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.savefig(fr"C:\Users\manat\project2\tmp_output\exp_yfs_{pitch}.eps", format="eps", dpi=300)
    plt.show()





def select_sim(pitch, depth_list):
    sim_data_ref = np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth0.csv")
    yf_ref, freq_ref = make_fftdata(kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(f"C:\\Users\\manat\\project2\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv")
        yf[i], freq[i] = make_fftdata(kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right), exp_dt)


    return freq, yf, yf_ref


def select_exp(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 0)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i], gomi = import_data(pitch, depth_list[i])
        yf[i], freq[i] = make_fftdata(kiritori2(sim_data[i], left, right), exp_dt)


    return freq, yf, yf_ref

def extract_frequency_band_2d(freq, fft_data_2d, freq_min, freq_max):
    """
    2次元スペクトルデータから特定の周波数帯のデータを抽出する関数
    
    Parameters:
    -----------
    freq : array-like
        周波数配列
    fft_data_2d : 2D array-like
        2次元振幅スペクトル配列（行が異なるスペクトル、列が周波数に対応）
    freq_min : float
        抽出したい周波数の下限
    freq_max : float
        抽出したい周波数の上限
    
    Returns:
    --------
    fft_band_2d : ndarray
        抽出された2次元振幅スペクトル配列
    freq_band : ndarray
        抽出された周波数配列
    """
    # 指定された周波数範囲のインデックスを取得
    mask = (freq >= freq_min) & (freq <= freq_max)
    
    # マスクを使って対応するデータを抽出（2次元データの場合は列方向にマスクを適用）
    freq_band = freq[mask]
    fft_band_2d = fft_data_2d[:, mask]
    
    return fft_band_2d, freq_band

def calculate_rmse_with_frequency_band(freq, array1, array2, freq_min, freq_max):
    """
    指定された周波数帯域でマスクをかけた後、2つの2次元NumPy配列の各行ごとにRMSEを計算する関数
    
    Parameters:
    -----------
    freq : array-like
        周波数配列
    array1 : 2D ndarray
        1つ目の2次元振幅スペクトル配列
    array2 : 2D ndarray
        2つ目の2次元振幅スペクトル配列（array1と同じ形状である必要があります）
    freq_min : float
        抽出したい周波数の下限
    freq_max : float
        抽出したい周波数の上限
    
    Returns:
    --------
    row_rmse : ndarray
        各行（各スペクトル）のRMSEを含む1次元配列
    masked_array1 : 2D ndarray
        マスク後の1つ目の配列
    masked_array2 : 2D ndarray
        マスク後の2つ目の配列
    freq_band : ndarray
        マスク後の周波数配列
    """
    # 配列の形状が一致しているか確認
    if array1.shape != array2.shape:
        raise ValueError("両方の配列は同じ形状である必要があります")
    
    # 周波数帯域でマスクをかける
    masked_array1, freq_band = extract_frequency_band_2d(freq, array1, freq_min, freq_max)
    masked_array2, _ = extract_frequency_band_2d(freq, array2, freq_min, freq_max)
    
    # 各行ごとにRMSEを計算
    squared_diff = np.square(masked_array1 - masked_array2)  # 差の二乗
    mean_squared_diff = np.mean(squared_diff, axis=1)  # 行ごとの平均二乗誤差
    row_rmse = np.sqrt(mean_squared_diff)  # 平方根を取る
    
    return row_rmse, masked_array1, masked_array2, freq_band

def calculate_column_statistics_with_frequency_band(freq, array1, array2, freq_min, freq_max):
    """
    指定された周波数帯域でマスクをかけた後、2つの2次元NumPy配列の列ごとの誤差統計を計算する関数
    
    Parameters:
    -----------
    freq : array-like
        周波数配列
    array1 : 2D ndarray
        1つ目の2次元振幅スペクトル配列
    array2 : 2D ndarray
        2つ目の2次元振幅スペクトル配列（array1と同じ形状である必要があります）
    freq_min : float
        抽出したい周波数の下限
    freq_max : float
        抽出したい周波数の上限
    
    Returns:
    --------
    stats : dict
        列ごとの誤差統計情報を含む辞書
    masked_array1 : 2D ndarray
        マスク後の1つ目の配列
    masked_array2 : 2D ndarray
        マスク後の2つ目の配列
    freq_band : ndarray
        マスク後の周波数配列
    """
    # 配列の形状が一致しているか確認
    if array1.shape != array2.shape:
        raise ValueError("両方の配列は同じ形状である必要があります")
    
    # 周波数帯域でマスクをかける
    masked_array1, freq_band = extract_frequency_band_2d(freq, array1, freq_min, freq_max)
    masked_array2, _ = extract_frequency_band_2d(freq, array2, freq_min, freq_max)
    
    # 列ごとのRMSEを計算（ここでの列は各周波数ポイントに対応）
    squared_diff = np.square(masked_array1 - masked_array2)
    mean_squared_diff = np.mean(squared_diff, axis=0)
    column_rmse = np.sqrt(mean_squared_diff)
    
    # 絶対誤差を計算
    abs_diff = np.abs(masked_array1 - masked_array2)
    
    # 列ごとの統計情報を計算
    stats = {
        'rmse': column_rmse,
        'max_error': np.max(abs_diff, axis=0),
        'min_error': np.min(abs_diff, axis=0),
        'mean_error': np.mean(abs_diff, axis=0),
        'median_error': np.median(abs_diff, axis=0),
        'std_error': np.std(abs_diff, axis=0)
    }
    
    # 最も誤差が大きい列（周波数ポイント）のインデックス
    max_rmse_index = np.argmax(column_rmse)
    stats['max_rmse_column_index'] = max_rmse_index
    stats['max_rmse_frequency'] = freq_band[max_rmse_index]
    
    return stats, masked_array1, masked_array2, freq_band



def make_fftdata_wavy(data, dt_sample, wave_speed=cl, wavenumber_type='standard'):
    """
    FFTを計算し、周波数または波数で結果を返す関数
    
    Parameters:
    -----------
    data : array_like
        FFT対象のデータ
    dt_sample : float
        サンプリング時間間隔
    wave_speed : float, optional
        波の速度（音波の場合は音速、デフォルトは空気中の音速343 m/s）
    wavenumber_type : str, optional
        'standard'（波数 = 1/λ）または'angular'（角波数 = 2π/λ）
        
    Returns:
    --------
    magnitude : ndarray
        FFT結果の絶対値
    k_or_freq : ndarray
        周波数または波数の軸データ
    """
    howmanyzero = fft_N - len(data)
    data_fft = np.concatenate([np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))], 0)
    yf = np.fft.fft(data_fft) / (fft_N / 2)
    
    # 周波数軸の計算
    freq = np.linspace(0, 1.0 / dt_sample, fft_N)
    
    # 周波数から波数への変換
    if wavenumber_type == 'angular':
        # 角波数 k = 2π/λ = 2πf/c [rad/m]
        k_values = 2 * np.pi * freq / wave_speed
    else:
        # 標準波数 k = 1/λ = f/c [1/m]
        k_values = freq / wave_speed
    
    return np.abs(yf), k_values


def select_pitch_and_image_wavy(pitch, depth_list):
    sim_data_ref = np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth0.csv")
    yf_ref, freq_ref = make_fftdata_wavy(kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(f"C:\\Users\\manat\\project2\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv")
        yf[i], freq[i] = make_fftdata(kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right), exp_dt)

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{int(depth_list[i])/100} mm")
    plt.xlim(2000000/cl, 8000000/cl)
    plt.ylim(0, 1.5)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Wave number m^-1")
    plt.legend()
    plt.tight_layout()
    plt.show()



def select_pitch_and_image_exp_wavy(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 0)
    yf_ref, freq_ref = make_fftdata_wavy(kiritori2(sim_data_ref, left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i], gomi = import_data(pitch, depth_list[i])
        yf[i], freq[i] = make_fftdata(kiritori2(sim_data[i], left, right), exp_dt)

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{depth_list[i]/100} mm")
    plt.xlim(2000000/cl, 8000000/cl)
    plt.ylim(0, 1.5)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Wave number m^-1")
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_fftdata_main_tale(data, dt, boader):
    yf,freq = np.zeros((3,fft_N)),np.zeros((3,fft_N))
    howmanyzero0 = fft_N-len(data)
    data_fft0 = np.concatenate([np.zeros(int(howmanyzero0 / 2)), data, np.zeros(int(howmanyzero0 / 2))], 0)
    yf0 = np.fft.fft(data_fft0) / (fft_N / 2)
    freq1 = np.linspace(0, 1.0 / dt, fft_N)
    howmanyzero1 = fft_N -len(data[:boader])
    data_fft1 = np.concatenate([np.zeros(int(howmanyzero1 / 2)), data[:boader], np.zeros(int(howmanyzero1 / 2))], 0)
    yf1 = np.fft.fft(data_fft1)/(fft_N/2)
    howmanyzero2 = fft_N -len(data[boader:])
    data_fft2 = np.concatenate([np.zeros(int(howmanyzero2 / 2)), data[boader:], np.zeros(int(howmanyzero2 / 2))], 0)
    yf2 = np.fft.fft(data_fft2)/(fft_N/2)
    yf[0],freq[0] = np.abs(yf0),freq1
    yf[1],freq[1] = np.abs(yf1),freq1
    yf[2],freq[2] = np.abs(yf2),freq1
    return yf,freq