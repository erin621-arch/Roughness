import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog

# ============================================================
#  基本パラメータ
# ============================================================

fft_N = 2 ** 14

x_length = 0.02       # [m]
y_length = 0.04       # [m]
mesh_length = 1.0e-5  # [m]

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P波速度
ct = np.sqrt(G / rho)                                    # S波速度

dx = x_length / int(x_length / mesh_length)

# FDTD dt
dt_sim = dx / cl / np.sqrt(6)
print(f"FDTD dt_sim = {dt_sim:.3e} [s]")

# ゲート幅（ピーク周り切り出し）
left  = 2 ** 10
right = (2 ** 10) * 3

# ゲート開始位置（シミュレーション用）
sim_gate_start = 7560

# ============================================================
#  ユーティリティ
# ============================================================

def kiritori2_safe(data_sample, left, right):
    """
    波形の最大値まわりに [left, right] だけ切り出し（範囲外は安全にクリップ）。
    """
    datamaxhere = np.nanargmax(data_sample)
    datamaxstart = max(0, datamaxhere - left)
    datamaxend   = min(len(data_sample), datamaxhere + right)
    return data_sample[datamaxstart:datamaxend], datamaxhere, datamaxstart, datamaxend

def interpolate_to_target_dt(data_raw, dt_original, dt_target):
    """波形 data_raw (dt_original 刻み) を dt_target 刻みに再補間する（1D）。"""
    t_original = np.arange(0, len(data_raw) * dt_original, dt_original)
    t_max = (len(data_raw) - 1) * dt_original
    t_new = np.arange(0, t_max, dt_target)
    func = interp1d(t_original, data_raw, kind='cubic')
    return func(t_new)

def get_exp_folder_path(base_root, shape, pitch, depth):
    """実験データのフォルダパスを自動生成"""
    if shape.lower() == "smooth":
        return os.path.join(base_root, "Smooth", "0_0")

    shape_map = {
        "sankaku": "Sankaku",
        "kusabi":  "Kusabi",
        "hanen":   "Hanen"
    }
    dir_shape = shape_map.get(shape, shape.capitalize())
    p_val = pitch.replace("p", "").replace(".", "")
    try:
        d_num = float(depth.replace("d", ""))
        val_100 = d_num * 100
        if val_100.is_integer():
            d_str = str(int(val_100))
        else:
            d_str = str(val_100)
    except ValueError:
        d_str = depth.replace("d", "").replace(".", "")

    dir_name = f"{p_val}_{d_str}"
    return os.path.join(base_root, dir_shape, dir_name)

def get_sim_folder_path(base_root, shape):
    """シミュレーションデータの格納フォルダ"""
    shape_map = {
        "sankaku": "Sankaku",
        "kusabi":  "Kusabi",
        "hanen":   "Hanen",
        "smooth":  "Smooth"
    }
    dir_shape = shape_map.get(shape, shape.capitalize())
    return os.path.join(base_root, dir_shape)

def load_experiment_time_axis(target_dir):
    """実験データから時間軸情報(dt)のみを取得するための簡易関数"""
    if not os.path.exists(target_dir):
        return None, None
    search_path = os.path.join(target_dir, "scope_*.csv")
    file_list = sorted(glob.glob(search_path))
    if not file_list:
        return None, None
    
    # 1つだけ読んでdtを取得
    try:
        data = np.genfromtxt(file_list[0], delimiter=",", skip_header=2)
        if data.ndim > 1:
            t = data[:, 0]
            dt = t[1] - t[0]
            return t, dt
    except:
        pass
    return None, None

# ============================================================
#  メイン処理
# ============================================================

if __name__ == "__main__":

    # --- ディレクトリ設定 ---
    # doc_path = r"C:/Users/hisay/OneDrive/ドキュメント/Test_folder"
    # doc_path = "/Users/hisayoshi/project_python/Roughness/Test_folder"
    doc_path = r"C:\Users\cs16\Roughness\project4"  # 研究室PC
    
    exp_base_dir = os.path.join(doc_path, "Experiment_Data")
    sim_base_dir = os.path.join(doc_path, "Simulation_Data")

    # 実験条件（dt_exp取得用）
    target_shape = "hanen"
    target_pitch = "p1.25"
    target_depth = "d0.20"
    
    print("-" * 60)
    print("処理を開始します...")

    # 1. dt_exp の決定（シミュレーションのリサンプリング用）
    try:
        target_exp_dir = get_exp_folder_path(exp_base_dir, target_shape, target_pitch, target_depth)
        _, dt_exp = load_experiment_time_axis(target_exp_dir)
    except Exception as e:
        dt_exp = None

    if dt_exp is None:
        print("Warning: 実験データからdtを取得できませんでした。dt_sim をそのまま使用します。")
        dt_exp = dt_sim
    else:
        print(f"Experimental dt detected: {dt_exp:.3e} [s]")

    # 2. ファイル選択ダイアログの表示 (複数選択可能)
    print("表示したいシミュレーションファイル(.csv)を選択してください...")
    root = tk.Tk()
    root.withdraw() # メインウィンドウを隠す
    
    try:
        initial_dir = get_sim_folder_path(sim_base_dir, target_shape)
        if not os.path.exists(initial_dir):
            initial_dir = sim_base_dir
    except:
        initial_dir = sim_base_dir

    filenames = filedialog.askopenfilenames(
        title="シミュレーションCSVファイルを選択（複数選択可）",
        initialdir=initial_dir,
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not filenames:
        print("ファイルが選択されませんでした。終了します。")
        exit()

    print(f"{len(filenames)} 個のファイルが選択されました。")

    # ★ 重ね書き用のグラフ（Figure）を事前に作成
    fig_all, ax_all = plt.subplots(figsize=(10, 6))

    # 3. ループ処理でグラフ描画
    for idx, fpath in enumerate(filenames):
        fname = os.path.basename(fpath)
        print(f"Processing: {fname}")

        try:
            # データの読み込み
            wave_sim = np.loadtxt(fpath, delimiter=",", dtype=float)
            
            # リサンプリング
            wave_sim_resampled = interpolate_to_target_dt(wave_sim, dt_sim, dt_exp)
            
            # ゲート処理
            tail_sim = wave_sim_resampled[sim_gate_start:]
            seg_sim, _, _, _ = kiritori2_safe(tail_sim, left, right)

            # 時間軸作成 [s]
            t_rel_sim = np.arange(len(seg_sim)) * dt_exp 

            # --------------------------------------------------------
            # A. 個別のグラフ描画 (ラベルなし)
            # --------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # プロット (正負反転)
            ax.plot(t_rel_sim, -seg_sim, color='tab:blue', linewidth=1.5)

            # 軸ラベル
            ax.set_xlabel("Time [s]", fontsize=12)
            ax.set_ylabel("Pressure [Pa]", color='black', fontsize=12)
            ax.set_title(f"Simulation Gated Waveform\n{fname}", loc='left', fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='black')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            # 縦軸範囲固定
            ax.set_ylim(-1, 1)

            # レイアウト
            plt.tight_layout()

            # --------------------------------------------------------
            # B. 重ね書き用のグラフへのプロット追加 (ラベルあり)
            # --------------------------------------------------------
            ax_all.plot(t_rel_sim, -seg_sim, linewidth=1.5, label=fname)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # --------------------------------------------------------
    # C. 重ね書きグラフの仕上げ
    # --------------------------------------------------------
    ax_all.set_xlabel("Time [s]", fontsize=12)
    ax_all.set_ylabel("Pressure [Pa]", color='black', fontsize=12)
    ax_all.set_title("Combined Simulation Waveforms", loc='left', fontsize=14, fontweight='bold')
    
    ax_all.tick_params(axis='y', labelcolor='black')
    ax_all.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # 縦軸範囲固定
    ax_all.set_ylim(-1, 1)

    # ★ 凡例（ラベル）を表示
    ax_all.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # すべてのグラフを同時に表示
    print("すべてのグラフを表示します...")
    plt.show()