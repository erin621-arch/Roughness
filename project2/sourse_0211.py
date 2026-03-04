import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

def create_wave_matrix(wave_list):
    """
    個別の波形データのリストから2次元配列を生成する
    
    Parameters:
    -----------
    wave_list : list of array-like
        波形データのリスト。各要素は1次元配列
    
    Returns:
    --------
    wave_matrix : 2D numpy array
        波形データを積み重ねた2次元配列
        shape = (パラメータの数, 最短の波形の長さ)
    """
    # 最も短い波形の長さを取得
    min_length = min(len(wave) for wave in wave_list)
    
    # 各波形を最短の長さに切り詰めて2次元配列に格納
    wave_matrix = np.array([wave[:min_length] for wave in wave_list])
    
    return wave_matrix

def calculate_peak_means(wave_matrix):
    """
    波形データの各パラメータごとにピークの絶対値の平均値を計算する
    正と負の両方のピークを検出し、その絶対値の平均を取る
    """
    peak_means = []
    peak_indices_list = []
    
    for wave in wave_matrix:
        # 正のピークを検出
        positive_peak_indices, _ = find_peaks(wave)
        # 負のピークを検出（波形を反転させて検出）
        negative_peak_indices, _ = find_peaks(-wave)
        
        # 両方のピークのインデックスを結合
        all_peak_indices = np.concatenate([positive_peak_indices, negative_peak_indices])
        all_peak_indices.sort()  # 時系列順にソート
        
        # ピーク値の絶対値を取得
        peak_values = np.abs(wave[all_peak_indices])
        
        # ピークの絶対値の平均値を計算
        if len(peak_values) > 0:
            peak_means.append(np.mean(peak_values))
        else:
            peak_means.append(np.nan)
        
        peak_indices_list.append(all_peak_indices)
    
    return np.array(peak_means), peak_indices_list


def calculate_absolute_means(wave_matrix):
    """
    波形データの各パラメータごとに絶対値の平均値を計算する
    
    Parameters:
    -----------
    wave_matrix : 2D array-like
        波形データの2次元配列。shape = (パラメータの数, 時間軸の長さ)
    
    Returns:
    --------
    abs_means : numpy array
        各パラメータの波形の絶対値の平均値
    """
    # 各波形の絶対値の平均を計算
    abs_means = np.mean(np.abs(wave_matrix), axis=1)
    return abs_means

def adjust_data_length(*wave_data_list):
    """
    複数の波形データの長さを最も短いものに合わせる
    
    Parameters:
    -----------
    *wave_data_list : 2D array-like
        波形データの2次元配列のリスト
    
    Returns:
    --------
    list of 2D array-like
        長さを調整した波形データのリスト
    """
    # 各波形データの時間軸長を取得
    lengths = [data.shape[1] for data in wave_data_list]
    
    # 最短の長さを取得
    min_length = min(lengths)
    
    # すべてのデータを最短の長さに切り詰める
    return [data[:, :min_length] for data in wave_data_list]

def create_wave_animation(*wave_data_list, param_values, dt, 
                         interval=500, figsize=(12, 6), show_peaks=True,
                         colors=['b', 'r', 'g'], labels=None):
    """
    複数の波形データとそれぞれのパラメータ値からアニメーションを作成する関数
    
    Parameters:
    -----------
    *wave_data_list : 2D array-like
        波形データの2次元配列のリスト
    param_values : array-like
        各波形データに対応するパラメータ値
    dt : float
        サンプリング間隔
    interval : int, optional
        アニメーションの更新間隔（ミリ秒）
    figsize : tuple, optional
        図のサイズ
    show_peaks : bool, optional
        ピークを表示するかどうか
    colors : list of str, optional
        各波形の色
    labels : list of str, optional
        各波形のラベル
    """
    # データの長さを調整
    wave_data_list = adjust_data_length(*wave_data_list)
    
    # ラベルがない場合はデフォルトのラベルを使用
    if labels is None:
        labels = [f'Wave {i+1}' for i in range(len(wave_data_list))]
    
    # ピークの位置を計算
    if show_peaks:
        peak_indices_list = [calculate_peak_means(data)[1] for data in wave_data_list]
    
    # 時間軸の生成
    time = np.arange(wave_data_list[0].shape[1]) * dt
    
    # データの検証
    for data in wave_data_list:
        if len(data) != len(param_values):
            raise ValueError("波形データとパラメータ値の数が一致していません")
    
    # プロットの初期設定
    fig, ax = plt.subplots(figsize=figsize)
    lines = []
    peaks = []
    
    for i, (data, color, label) in enumerate(zip(wave_data_list, colors, labels)):
        # 波形のライン
        line, = ax.plot([], [], f'{color}-', label=label)
        lines.append(line)
        
        # ピークのマーカー
        if show_peaks:
            peak = ax.plot([], [], f'{color}o', label=f'{label} Peaks', markersize=8)[0]
            peaks.append(peak)
    
    param_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # グラフの見た目を設定
    ax.set_xlim(time[0], time[-1])
    y_max = max(np.max(np.abs(data)) for data in wave_data_list) * 1.1
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    
    def init():
        """アニメーションの初期化"""
        for line in lines:
            line.set_data([], [])
        if show_peaks:
            for peak in peaks:
                peak.set_data([], [])
        
        artists = lines + ([param_text] if not show_peaks else peaks + [param_text])
        return artists
    
    def animate(frame):
        """各フレームの更新"""
        artists = []
        
        # 波形の更新
        for i, (line, data) in enumerate(zip(lines, wave_data_list)):
            line.set_data(time, data[frame])
            artists.append(line)
            
            # ピークの更新
            if show_peaks:
                peaks[i].set_data(time[peak_indices_list[i][frame]], 
                                data[frame][peak_indices_list[i][frame]])
                artists.append(peaks[i])
        
        param_text.set_text(f'Parameter: {param_values[frame]:.2f}')
        artists.append(param_text)
        
        return artists
    
    # アニメーションの作成
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(param_values),
                        interval=interval, blit=True)
    
    return anim

def create_static_comparison(*wave_data_list, param_values, dt,
                           num_plots=6, figsize=(12, 15),
                           colors=['b', 'r', 'g'], labels=None):
    """
    パラメータを変えた時の波形の比較を静的なプロットで表示
    """
    # データの長さを調整
    wave_data_list = adjust_data_length(*wave_data_list)
    
    # ラベルがない場合はデフォルトのラベルを使用
    if labels is None:
        labels = [f'Wave {i+1}' for i in range(len(wave_data_list))]
    
    # 時間軸の生成
    time = np.arange(wave_data_list[0].shape[1]) * dt
    
    # 表示するインデックスを選択
    if num_plots > len(param_values):
        num_plots = len(param_values)
    indices = np.linspace(0, len(param_values)-1, num_plots, dtype=int)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    if num_plots == 1:
        axes = [axes]
    fig.suptitle('Wave Amplitude Comparison for Different Parameters')
    
    # 最初のサブプロットで凡例用のラインを作成
    lines = []
    for ax, idx in zip(axes, indices):
        temp_lines = []
        for data, color, label in zip(wave_data_list, colors, labels):
            line, = ax.plot(time, data[idx], f'{color}-')
            temp_lines.append(line)
        if ax == axes[0]:  # 最初のプロットのラインを保存
            lines = temp_lines
        ax.grid(True)
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-0.9, 0.9)
        ax.set_title(f'Parameter = {param_values[idx]:.2f}')
    
    # 図全体で1つの凡例を作成（グラフの右側）
    fig.legend(lines, labels, bbox_to_anchor=(1.15, 0.5), loc='center left')
    
    axes[-1].set_xlabel('Time [s]')
    # 凡例のスペースを確保するために右側に余白を追加
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    return fig