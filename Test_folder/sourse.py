import numpy as np
from scipy.interpolate import interp1d
import re
import pandas as pd
import matplotlib.pyplot as plt

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
left = 2 ** 10
right = (2 ** 10) * 3
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
    datamaxhere = np.nanargmax(data_sumple)
    datamaxstart = datamaxhere - left
    datamaxend = datamaxhere + right
    return data_sumple[datamaxstart:datamaxend]


def make_fftdata(data, dt_sumple):
    howmanyzero = fft_N - len(data)
    data_fft = np.concatenate([np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))], 0)
    yf = np.fft.fft(data_fft) / (fft_N / 2)
    freq = np.linspace(0, 1.0 / dt_sumple, fft_N)
    return np.abs(yf), freq

def make_fftdata_tale(data, dt_sumple, kugiristeppulsekara):
    howmanyzero = fft_N - len(data)
    data_fft = np.concatenate([np.zeros(int(howmanyzero / 2)), data, np.zeros(int(howmanyzero / 2))], 0)
    yf = np.fft.fft(data_fft) / (fft_N / 2)
    freq = np.linspace(0, 1.0 / dt_sumple, fft_N)

    data_tale_howmanyzero = fft_N - len(data[kugiristeppulsekara:])
    data_tale = np.concatenate([np.zeros(int(data_tale_howmanyzero / 2)), data[kugiristeppulsekara:], np.zeros(int(data_tale_howmanyzero / 2))])
    yf_tale = np.fft.fft(data_tale) / (fft_N / 2)
    return np.abs(yf), np.abs(yf_tale), freq


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



def whererough(pitch, rough):
    match pitch:
        case 0:
            return 0, 1
        case 1250:
            if rough < 2:
                return pitch_125[rough], 2
            else:
                return pitch_125[rough], 1
        case 2000:
            if rough < 3:
                return pitch_200[rough], 2
            else:
                return pitch_200[rough], 1
        case 1500:
            if rough < 3:
                return pitch_150[rough], 3
            else:
                return pitch_150[rough], 2
        case 1750:
            if rough < 3:
                return pitch_175[rough], 3
            else:
                return pitch_175[rough], 2
        case _:
            return 0


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


def import_data(where, each, mode):
    data_tmp0 = []
    data_tmp1 = []
    data_tmp2 = []
    data_tmp3 = []
    data_tmp4 = []
    number_max = []
    data = []
    print(f'importing : C:\\Users\\Fujii Kotaro\\project1\\experience_data\\scope_5m_{mode}s_{where}.csv')
    for i in range(each):
        data_tmp0.append(
            pd.read_csv(f'C:\\Users\\Fujii Kotaro\\project1\\experience_data\\scope_5m_{mode}s_{i + where}.csv',
                        encoding='SHIFT-JIS', header=None))
        
        data_tmp1.append(data_tmp0[i].drop(data_tmp0[i].index[[0, 1]]))
        data_tmp1[i] = data_tmp1[i].astype(float)
        data_tmp2.append(data_tmp1[i].iloc[:, 0])
        data_tmp3.append(data_tmp1[i].iloc[:, 1])
        number_max.append(data_tmp2[i].count())
        data_tmp4.append([data_tmp3[i].mode().iloc[0]] * number_max[i])
        data.append(data_tmp3[i] - data_tmp4[i])
    data_arr = np.array(data)
    data_m = data_arr.mean(0)
    return data_m, data_arr


def make_dataset_exp_y_new(pitch, rough):
    depth_set = np.full(1, float(depthrough(pitch, rough)))
    pitch_set = np.full(1, float(pitch))
    return np.column_stack((pitch_set/10, depth_set))


def make_dataset_exp_new(pitch, rough, a1, a2, a3):
    mean, data_set_raw_exp = import_data(whererough(pitch, rough)[0], 10, whererough(pitch, rough)[1])
    mean0, data_set_raw_exp0 = import_data(0, 10, 1)
    data_test_exp, freq = make_data(mean0, data_set_raw_exp[0], a1)
    data_set_exp = np.zeros((1, len(data_test_exp)))
    data_set_exp[0], freq = make_data(mean0, mean, a1)
    return data_set_exp


def image_many(zenbu,tale,freq):
    fig,ax = plt.subplots(1,1,figsize=(8,6),sharey = True)
    ax.plot(freq,zenbu,label = 'Whole')
    ax.plot(freq,tale,label = 'Only tale')
    ax.set_xlim(0,10000000)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(0, 0.03)
    plt.legend()

def image_many_all(zenbu, main, tale, freq):
    fig,ax = plt.subplots(1,1,figsize=(8,6),sharey = True)
    ax.plot(freq,zenbu,label = 'Whole')
    ax.plot(freq, main, label= "Main")
    ax.plot(freq,tale,label = 'Only tale')
    ax.set_xlim(0,10000000)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(0, zenbu.max()*1.25)
    plt.legend()

def image_ref(zenbu,freq):
    fig,ax = plt.subplots(1,1,figsize=(8,6),sharey = True)
    ax.plot(freq,zenbu,label = 'Whole')
    ax.set_xlim(0,10000000)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(0, 0.03)

def image_wave(data, dt):
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    time = np.arange(0, len(data)*dt, dt)
    ax.plot(time, data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_ylim(-0.7, 1.0)


def image_many_sim(zenbu,tale,freq):
    fig,ax = plt.subplots(1,1,figsize=(8,6),sharey = True)
    ax.plot(freq,zenbu,label = 'Whole')
    ax.plot(freq,tale,label = 'Only tale')
    ax.set_xlim(0,10000000)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(0, 0.06)
    plt.legend()

def image_ref_sim(zenbu,freq):
    fig,ax = plt.subplots(1,1,figsize=(8,6),sharey = True)
    ax.plot(freq,zenbu,label = 'Whole')
    ax.set_xlim(0,10000000)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(0, 0.06)

def image_wave_sim(data, dt):
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    time = np.arange(0, len(data)*dt, dt)
    ax.plot(time, data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_ylim(-1.7, 1.5)

def select_pitch_and_image(pitch, depth_list):
    sim_data_ref = np.loadtxt(r"C:\Users\Fujii Kotaro\project1\data_all\cupy_pitch125_depth0.csv")
    yf_ref, freq_ref = make_fftdata(kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(f"C:\\Users\\Fujii Kotaro\\project1\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv")
        yf[i], freq[i] = make_fftdata(kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right), exp_dt)

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{int(depth_list[i])/100} mm")
    plt.xlim(2000000, 8000000)
    plt.ylim(0, 1.5)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Frequency Hz")
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_pitch_and_image_exp(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 10, 1)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
    
        where, rough = whererough(int(pitch*10), 
                           depth_index_all[int(pitch*10)].index(depth_list[i]))
        sim_data[i], gomi = import_data(where, 10, rough)
        yf[i], freq[i] = make_fftdata(kiritori2(sim_data[i], left, right), exp_dt)

    plt.rcParams["font.size"] = 14
    plt.figure()
    for i in range(len(depth_list)):
        plt.plot(freq_ref, yf[i] / yf_ref, label=f"d:{depth_list[i]/100} mm")
    plt.xlim(2000000, 8000000)
    plt.ylim(0, 1.5)
    plt.ylabel("Normalized amplitude")
    plt.xlabel("Frequency Hz")
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_pitch(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 10, 1)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
    
        where, rough = whererough(int(pitch*10), 
                           depth_index_all[int(pitch*10)].index(depth_list[i]))
        sim_data[i], gomi = import_data(where, 10, rough)
        yf[i], freq[i] = make_fftdata(kiritori2(sim_data[i], left, right), exp_dt)

    return yf, freq[0]


def select_exp_data(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 10, 1)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    
    for i in range(len(depth_list)):
    
        where, rough = whererough(int(pitch*10), 
                           depth_index_all[int(pitch*10)].index(depth_list[i]))
        sim_data[i], gomi = import_data(where, 10, rough)

    return sim_data


def select_sim(pitch, depth_list):
    sim_data_ref = np.loadtxt(r"C:\Users\Fujii Kotaro\project1\data_all\cupy_pitch125_depth0.csv")
    yf_ref, freq_ref = make_fftdata(kiritori2(interpolate_sim_one(sim_data_ref)[8000:], left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
        sim_data[i] = np.loadtxt(f"C:\\Users\\Fujii Kotaro\\project1\\data_all\\cupy_pitch{pitch}_depth{depth_list[i]}.csv")
        yf[i], freq[i] = make_fftdata(kiritori2(interpolate_sim_one(sim_data[i])[8000:], left, right), exp_dt)


    return freq, yf, yf_ref


def select_exp(pitch, depth_list):
    sim_data_ref, gomi = import_data(0, 10, 1)
    yf_ref, freq_ref = make_fftdata(kiritori2(sim_data_ref, left, right), exp_dt)
    
    sim_data = np.zeros((len(depth_list), len(sim_data_ref)))
    yf = np.zeros((len(depth_list), len(yf_ref)))
    freq = np.zeros((len(depth_list), len(freq_ref)))
    
    for i in range(len(depth_list)):
    
        where, rough = whererough(int(pitch*10), 
                           depth_index_all[int(pitch*10)].index(depth_list[i]))
        sim_data[i], gomi = import_data(where, 10, rough)
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