import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time

# ================== 調整パラメータ ==================

# 出力先ディレクトリ
# output_dir = r"tmp_output" 
output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"  # 自宅PC等のパスに合わせてください

# きずパラメータ
f_width = 0.25e-3   # 幅 [m]
f_pitch = 2.00e-3   # ピッチ [m]
f_depth = 0.20e-3   # 深さ [m]

# =======================================================================

# ---------------- 基本パラメータ ----------------
x_length = 0.02   # x方向の長さ [m]
y_length = 0.04   # y方向の長さ [m]
mesh_length = 1.0e-5  # メッシュ長 [m]

nx = int(x_length / mesh_length)  # how many mesh
ny = int(y_length / mesh_length)

dx = x_length / nx  # mesh length m
dy = y_length / ny  # m

rho = 7840   # density kg/m^3
E = 206 * 1e9  # young percentage kg/ms^2
G = 80 * 1e9   # stiffness
V = 0.27   # poisson ratio

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P wave
ct = np.sqrt(G / rho)   # S wave
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)  # time mesh
f = 4.7e6   # frequency 
T = 1 / f   # period
lam = cl / f   # lambda
k = 1 / lam   # wave number
n = T / dt   # 波が離散点上で何点か

# ---------------- きず生成関数  ----------------
def isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length):
    # 1:通常 / 0:自由境界（ゼロにする領域）
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree = np.ones((nx + 1, ny + 1))
    
    # 離散化
    mn_w = int(f_width / mesh_length)   # きずの幅の離散点数
    mn_p = int(f_pitch / mesh_length)   # 1ピッチの離散点数
    mn_nf = int((f_pitch - f_width) / mesh_length)  # きずのない部分の離散点数
    mn_d = int(f_depth / mesh_length)   # きずの深さ方向の離散点数
    num_f = int(ny * mesh_length / f_pitch)   # きずの数
    
    # 外枠
    T13_isfree[0, 0:ny] = 0
    T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0] = 0
    T5_isfree[0:nx, ny] = 0
    T5_isfree[nx, 0:ny + 1] = 0

    # きず部分の自由面をつくる
    for i in range(num_f):
        if (i + 1) * mn_p >= ny:
            break
        # きず部分ゆえに消すとこ（矩形欠陥の処理を行うところ）
        T5_isfree[nx - mn_d:nx, mn_nf + i * mn_p:(i + 1) * mn_p] = 0
        T13_isfree[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0
        
    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 ----------------
def around_free():
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

    # Ux セル
    for i in range(nx):
        for j in range(ny):
            if T13_isfree[i, j] == 0:
                Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0:
                Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0:
                Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0:
                Ux_free_count[i, j] += 1

    # Uy ノード
    for i in range(nx + 1):
        for j in range(ny + 1):
            if i == 0 or (i == nx and j == 0) or (i == nx and j == ny):
                Uy_free_count[i, j] += 1
            if j == 0 or j == ny:
                Uy_free_count[i, j] += 1
            elif 0 < i and 0 < j:
                if T13_isfree[i, j - 1] == 0:
                    Uy_free_count[i, j] += 1
                if T13_isfree[i, j] == 0:
                    Uy_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0:
                    Uy_free_count[i, j] += 1
                if T5_isfree[i, j] == 0:
                    Uy_free_count[i, j] += 1
    return Ux_free_count, Uy_free_count


# ---------------- 入射波形 ----------------
wn = 2.5  # 波数
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3

# ---------------- 音源・計測設定 ----------------
sy = int(ny / 2)
sx = 0
# 探触子の直径 m
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)

t_max = 4 * x_length / cl / dt  # 1往復ちょいの時間

# ---------------- きずパラメータの離散値 ----------------
print(f"f_pitch = {f_pitch}")
print(f"f_depth = {f_depth}")

# きずの数
num_f = int(y_length / f_pitch)
# きずの幅の離散点数
mn_w = int(f_width / mesh_length)
# 1ピッチの離散点数
mn_p = int(f_pitch / mesh_length)
# きずのない部分の離散点数
mn_nf = int((f_pitch - f_width) / mesh_length)
# きずの深さ方向の離散点数
mn_d = int(f_depth / mesh_length)

# ---------------- FDTD配列 ----------------
T1 = cp.zeros((nx + 1, ny), dtype=float)
T3 = cp.zeros((nx + 1, ny), dtype=float)
T5 = cp.zeros((nx, ny + 1), dtype=float)
Ux = cp.zeros((nx, ny), dtype=float)
Uy = cp.zeros((nx + 1, ny + 1), dtype=float)

dtx = dt / dx
dty = dt / dy

# 初期化
T13_isfree, T5_isfree = isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length)
Ux_free_count, Uy_free_count = around_free()

wave = np.zeros(int(t_max))

start_time = time.time()

# ---------------- メイン時間ループ ----------------
for t in range(int(t_max)):
    # 進捗表示
    if t % 500 == 0:
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    # (5) 入射波 
    if t < int(len(wave4)):
        T1[0, sy_l:sy_r] = wave4[t]

    # 入射終了後の処理
    if t >= int(len(wave4)):
        Uy[0, sy_l:sy_r] = 0
        Ux[0, sy_l:sy_r] = 0
        T1[0, 0:ny] = 0
        T5[0, 0:ny] = 0

    # (1) 静的境界（応力の境界条件）
    T5[0:nx, 0] = 0
    T5[0:nx, ny] = 0
    T3[0, 0:ny] = 0
    T1[nx, 0:ny] = 0
    T3[nx, 0:ny] = 0
    T1[0, 0] = 0
    T3[0, 0] = 0
    T5[0, 0] = 0

    # (2) Uy の境界条件（横粒子速度の境界条件）
    Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
    Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
    Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])

    # (4) きず部分の応力境界条件 (Code A独自: for文ループ処理)
    for i in range(num_f):
        if (i + 1) * mn_p >= ny:
            break
        # きず部分ゆえに消すとこ
        T5[nx - mn_d:nx, mn_nf + i * mn_p:(i + 1) * mn_p] = 0
        T1[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0
        T3[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0

    # 重複適用されているがCode Aのまま維持
    # 横粒子速度の境界条件 (再)
    Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
    Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
    Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])

    # (3) 応力の更新
    T1[1:nx, 0:ny] = T1[1:nx, 0:ny] - dtx * ((c11 * (Ux[1:nx, 0:ny] - Ux[0:nx - 1, 0:ny]))
                                            + (c13 * (Uy[1:nx, 1:ny + 1] - Uy[1:nx, 0:ny])))

    T3[1:nx, 0:ny] = T3[1:nx, 0:ny] - dtx * ((c13 * (Ux[1:nx, 0:ny] - Ux[0:nx - 1, 0:ny]))
                                            + (c11 * (Uy[1:nx, 1:ny + 1] - Uy[1:nx, 0:ny])))

    T5[0:nx, 1:ny] = T5[0:nx, 1:ny] - dtx * c55 * (Ux[0:nx, 1:ny] - Ux[0:nx, 0:ny - 1]
                                                    + Uy[1:nx + 1, 1:ny] - Uy[0:nx, 1:ny])
    
    # (4) きず部分の応力境界条件 (再適用: Code A独自)
    for i in range(num_f):
        if (i + 1) * mn_p >= ny:
            break
        # きず部分ゆえに消すとこ
        T5[nx - mn_d:nx, mn_nf + i * mn_p:(i + 1) * mn_p] = 0
        T1[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0
        T3[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0

    # 重複適用されているがCode Aのまま維持
    # 横粒子速度の境界条件 (再)
    Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
    Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
    Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])

    # (6) 粒子速度の更新
    # 自由境界に接するノードと接しないノードを組み合わせて粒子速度を更新
    Ux[0:nx, 0:ny] = cp.where(cp.asarray(Ux_free_count[0:nx, 0:ny]) < 4,
                              Ux[0:nx, 0:ny] - (4 / rho / (4 - cp.asarray(Ux_free_count[0:nx, 0:ny]))) * dtx
                              * (T1[1:nx + 1, 0:ny] - T1[0:nx, 0:ny]
                                 + T5[0:nx, 1: ny + 1] - T5[0:nx, 0: ny]), 0)

    Uy[1:nx, 1:ny] = cp.where(cp.asarray(Uy_free_count[1:nx, 1:ny]) < 4,
                              Uy[1:nx, 1:ny] - (4 / rho / (4 - cp.asarray(Uy_free_count[1:nx, 1:ny]))) * dty
                              * (T3[1:nx, 1:ny] - T3[1:nx, 0:ny - 1]
                                 + T5[1:nx, 1:ny] - T5[0:nx - 1, 1:ny]), 0)
    
    cp.cuda.Device().synchronize()
    
    # (7) 記録
    if t > 0:
        wave[t] = cp.mean(T1[1, sy_l:sy_r])
        
        # スナップショット保存用（必要に応じて使用）
        if t == 5010:
            A = cp.asnumpy(T1)
        if t == 5687:
            B = cp.asnumpy(T1)
        if t == 6693:
            C = cp.asnumpy(T1)
        if t == 7298:
            D = cp.asnumpy(T1)

wave = cp.asnumpy(wave)
end_time = time.time()

print(f"f_pitch = {f_pitch}")
print(f"f_depth = {f_depth}")
print(f"実行時間: {end_time - start_time:.2f} 秒")

# ---------------- 出力先ディレクトリ準備 ----------------
os.makedirs(output_dir, exist_ok=True)

# ---------------- CSV保存 ----------------
# ファイル名ルールはCode Aのものを使用
csv_filename = f"cupy_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}.csv"
csv_path = os.path.join(output_dir, csv_filename)
np.savetxt(csv_path, wave, delimiter=',')
print(f"保存しました → {csv_path}")


# ---------------- 反射波形プロット ----------------
wave_data = wave # 既にnumpy配列
time_axis = np.arange(len(wave_data))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, wave_data, label='T1 average')
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.title(f"Waveform (pitch={f_pitch}, depth={f_depth})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()