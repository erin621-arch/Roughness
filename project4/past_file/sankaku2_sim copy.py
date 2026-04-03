import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time

# ================== 調整パラメータ ==================

# 出力先（研究室 / 自宅で切り替え）
# output_dir = r"C:/Users/cs16/Documents/Test_folder/tmp_output"  # 研究室PC
output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"   # 自宅PC

# きずパラメータ
f_width = 0.25e-3   # 幅 [m] (固定)
f_pitch = 1.25e-3   # ピッチ [m]
f_depth = 0.20e-3   # 深さ [m]

# ★追加パラメータ: 階段近似のブロックサイズ (1なら従来通り, 2なら2x2, 3なら3x3相当のブロック段差)
step_size = 1

# =======================================================================

# ---------------- 基本パラメータ ----------------
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

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P波
ct = np.sqrt(G / rho)                                     # S波
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)  # 時間刻み
f = 4.7e6                  # 周波数
T = 1 / f
lam = cl / f
k = 1 / lam
n = T / dt

# ---------------- 三角きず生成 ----------------
def isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length, step_size):
    # 1:通常 / 0:自由境界（ゼロにする領域）
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree  = np.ones((nx, ny + 1))

    # 離散化
    mn_w  = max(1, int(round(f_width  / mesh_length))) # 幅 W
    mn_p_val  = max(1, int(round(f_pitch  / mesh_length))) # 図面の p 値
    
    # すきま = p - 2W
    mn_nf = max(0, mn_p_val - 2 * mn_w)
    
    # 繰り返しの間隔（Period） = W + すきま
    mn_period = mn_w + mn_nf

    # 外枠
    T13_isfree[0, 0:ny]  = 0
    T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, ny]  = 0

    # きずの数（繰り返し間隔で割る）
    num_f = int(np.ceil(ny / mn_period))

    for i in range(num_f):
        # y_start = mn_nf + i * mn_period  # すきまスタートの場合
        y_start = i * mn_period          # きずスタートの場合（図面の左端基準ならこちら）
        
        if y_start >= ny:
            break
            
        y_end = min(y_start + mn_w, ny) # 幅 mn_w 分だけ確保
        y_center = (y_start + y_end) // 2

        # 深さ方向のループ
        mn_d  = max(1, int(round(f_depth  / mesh_length)))
        
        for d in range(mn_d):
            xi = (nx - 1) - d
            if xi < 0: break

            # ★変更箇所: step_size に基づいて深さをブロック化する
            # d を step_size で割った商を使って、Nマス分同じ幅を維持する
            d_step = (d // step_size) * step_size

            # 幅の計算に d ではなく d_step を使用する
            width_at_d = int(round(mn_w * (1.0 - (d_step + 0.5) / mn_d)))
            
            if width_at_d < 1: continue

            half = width_at_d // 2
            yl = max(y_center - half, 0)
            yr = min(y_center + half + (width_at_d % 2), ny)
            if yl >= yr: continue

            T5_isfree[xi, yl:yr] = 0
            if xi < nx + 1:
                T13_isfree[xi, yl:yr] = 0

    return T13_isfree, T5_isfree

# ---------------- 自由境界近傍の設定 ----------------
def around_free():
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

    # Ux セル（nx,ny）…周囲の4エッジ: T13(i,j), T13(i+1,j), T5(i,j), T5(i,j+1)
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

    # Uy ノード（nx+1,ny+1）…周囲の4エッジ: T13(i,j-1), T13(i,j), T5(i-1,j), T5(i,j)
    for i in range(nx + 1):
        for j in range(ny + 1):
            # 外枠の安全カウント
            if j == 0 or j == ny:
                Uy_free_count[i, j] += 1
            if i == 0 or i == nx:
                Uy_free_count[i, j] += 1

            # 内部のみ、周囲の自由エッジ数を数える
            if 0 < i < nx and 0 < j < ny:
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
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)

t_max = 4 * x_length / cl / dt  # 1往復ちょい

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

wave = np.zeros(int(t_max))

dtx = dt / dx
dty = dt / dy

# 初期化
T13_isfree, T5_isfree = isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length, step_size)
Ux_free_count, Uy_free_count = around_free()

start_time = time.time()

# ---------------- メイン時間ループ ----------------
for t in range(int(t_max)):
    # 進捗表示 (適宜コメントアウトしてもOK)
    if t % 500 == 0:
        print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

    # (1) 静的境界
    T5[0:nx, 0] = 0
    T5[0:nx, ny] = 0
    T3[0, 0:ny] = 0
    T1[nx, 0:ny] = 0
    T3[nx, 0:ny] = 0
    T1[0, 0] = 0
    T3[0, 0] = 0
    T5[0, 0] = 0

    # (2) Uy の境界条件
    Uy[1:nx, 0]  = Uy[1:nx, 0]  - (4 / rho) * dtx * (T3[1:nx, 0])
    Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
    Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])

    # (3) 応力更新
    T1[1:nx, 0:ny] = T1[1:nx, 0:ny] - dtx * (
        (c11 * (Ux[1:nx, 0:ny] - Ux[0:nx - 1, 0:ny])) +
        (c13 * (Uy[1:nx, 1:ny + 1] - Uy[1:nx, 0:ny]))
    )
    T3[1:nx, 0:ny] = T3[1:nx, 0:ny] - dtx * (
        (c13 * (Ux[1:nx, 0:ny] - Ux[0:nx - 1, 0:ny])) +
        (c11 * (Uy[1:nx, 1:ny + 1] - Uy[1:nx, 0:ny]))
    )
    T5[0:nx, 1:ny] = T5[0:nx, 1:ny] - dtx * c55 * (
        Ux[0:nx, 1:ny] - Ux[0:nx, 0:ny - 1] +
        Uy[1:nx + 1, 1:ny] - Uy[0:nx, 1:ny]
    )

    # (4) 三角型の自由面を応力に適用
    T1[T13_isfree == 0] = 0.0
    T3[T13_isfree == 0] = 0.0
    T5[T5_isfree[0:nx, :] == 0] = 0.0

    # (5) 音源注入
    if t < int(len(wave4)):
        T1[0, sy_l:sy_r] = wave4[t]
    else:
        # 参考コード準拠：固定端(U=0)と自由端(T=0)の混合境界
        Uy[0, sy_l:sy_r] = 0
        Ux[0, sy_l:sy_r] = 0
        T1[0, 0:ny] = 0
        T5[0, 0:ny] = 0

    # (6) 粒子速度更新
    Ux[0:nx, 0:ny] = cp.where(
        cp.asarray(Ux_free_count[0:nx, 0:ny]) < 4,
        Ux[0:nx, 0:ny] - (4 / rho / (4 - cp.asarray(Ux_free_count[0:nx, 0:ny]))) * dtx * (
            T1[1:nx + 1, 0:ny] - T1[0:nx, 0:ny] +
            T5[0:nx, 1:ny + 1] - T5[0:nx, 0:ny]
        ),
        0
    )
    Uy[1:nx, 1:ny] = cp.where(
        cp.asarray(Uy_free_count[1:nx, 1:ny]) < 4,
        Uy[1:nx, 1:ny] - (4 / rho / (4 - cp.asarray(Uy_free_count[1:nx, 1:ny]))) * dty * (
            T3[1:nx, 1:ny] - T3[1:nx, 0:ny - 1] +
            T5[1:nx, 1:ny] - T5[0:nx - 1, 1:ny]
        ),
        0
    )

    cp.cuda.Device().synchronize()

    # (7) 記録
    if t > 0:
        wave[t] = cp.mean(T1[1, sy_l:sy_r])

wave = cp.asnumpy(wave)
end_time = time.time()

print(f"f_pitch = {f_pitch}")
print(f"f_depth = {f_depth}")
print(f"実行時間: {end_time - start_time:.2f} 秒")

# ---------------- 出力先ディレクトリ準備 ----------------
os.makedirs(output_dir, exist_ok=True)

# ---------------- CSV保存 ----------------
# ファイル名に step_size (ブロックサイズ) を追加しました
csv_filename = f"sankaku2_cupy_pitch{int(f_pitch * 1e5)}_depth{int(f_depth * 1e5)}_blocksize={step_size}.csv"
csv_path = os.path.join(output_dir, csv_filename)
np.savetxt(csv_path, wave, delimiter=',')
print(f"保存しました → {csv_path}")

# ---------------- 反射波形プロット ----------------
wave_data = np.loadtxt(csv_path, delimiter=',')
time_axis = np.arange(len(wave_data))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, wave_data, label=f'T1 average (x=1, step={step_size})')
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.title(f"sankaku_Waveform at Probe Location (step={step_size})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()