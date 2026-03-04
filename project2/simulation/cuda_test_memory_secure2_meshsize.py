##########################################################################
#  シミュレーションを傷のバリエーション全部でやるやつ

import numpy as np
import cupy as cp
import gc

x_length = 0.02  # x方向の長さ m
y_length = 0.04  # y方向の長さ m
mesh_length = 0.5e-5  # m
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
f = 4.7e6  # frequency
T = 1 / f  # period
lam = cl / f  # lambda
k = 1 / lam  # wave number
n = T / dt  # 波が離散点上で何点か

# GPUメモリの明示的な解放を行うよう修正
def clear_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
    cp.cuda.Device().synchronize()


def isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length):
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree = np.ones((nx + 1, ny + 1))
    mn_w = int(f_width / mesh_length)  # きずの幅の離散点数
    mn_p = int(f_pitch / mesh_length)  # 1ピッチの離散点数
    mn_nf = int((f_pitch - f_width) / mesh_length)  # きずのない部分の離散点数
    mn_d = int(f_depth / mesh_length)  # きずの深さ方向の離散点数
    num_f = int(ny * mesh_length / f_pitch)  # きずの数
    T13_isfree[0, 0:ny] = 0
    T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0] = 0
    T5_isfree[0:nx, ny] = 0
    T5_isfree[nx, 0:ny + 1] = 0

    for i in range(num_f):
        if (i + 1) * mn_p >= ny:
            break
        #  きず部分ゆえに消すとこ
        T5_isfree[nx - mn_d:nx, mn_nf + i * mn_p:(i + 1) * mn_p] = 0
        T13_isfree[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0
    return T13_isfree, T5_isfree
def monitor_memory():
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    used_mem = total_mem - free_mem
    print(f"GPU Memory: Used {used_mem/1024**2:.1f}MB / Total {total_mem/1024**2:.1f}MB")

def around_free():
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

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


# =========================================
# 入射波の設定
wn = 2.5  # 波数
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3
# ==========================================

# 音源の位置
sy = int(ny / 2)
sx = 0
# 探触子の直径 m
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)

t_max = 4 * x_length / cl / dt  # 1往復ちょいの時間

#  =====================================================================================
# depth値（5刻みの間を埋める）
new_depth_values = [
    # 2, 3, 4, 5,      # 1-6の間
    # 7, 8, 9, 10,     # 6-11の間
    # 12, 
    # 13, 14, 15,  # 11-16の間
    # 17, 18, 19, 20,  # 16-21の間
    10   # 21-26の間
]
new_pitch_values = [ 
    200
]
print(new_pitch_values)
for omg in new_pitch_values:
    for j in new_depth_values:
        try:
            monitor_memory()
            wave = np.zeros(int(t_max))
            # 傷の幅 m
            f_width = 0.25e-3  # m
            # 傷の間隔 m
            f_pitch = omg*0.00001
            # 傷の深さ m
            f_depth = j*0.00001  # m

            print(f"f_pitch = {f_pitch}")
            print(f"f_depth = {f_depth}")
            #  傷の数
            num_f = int(y_length / f_pitch)
            #  傷の幅の離散点数
            mn_w = int(f_width / mesh_length)
            #  1ピッチの離散点数
            mn_p = int(f_pitch / mesh_length)
            # 傷のない部分の離散点数
            mn_nf = int((f_pitch - f_width) / mesh_length)
            # 傷の深さ方向の離散点数
            mn_d = int(f_depth / mesh_length)
            #  =========================================================================================
            #  =======FDTD本体=======
            # 　T1,T3は垂直応力 T5はせん断応力　Ux,Uyは固体粒子速度
            T1 = cp.zeros((nx + 1, ny), dtype=float)
            T3 = cp.zeros((nx + 1, ny), dtype=float)
            T5 = cp.zeros((nx, ny + 1), dtype=float)
            Ux = cp.zeros((nx, ny), dtype=float)
            Uy = cp.zeros((nx + 1, ny + 1), dtype=float)

            dtx = dt / dx
            dty = dt / dy

            T13_isfree, T5_isfree = isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length)
            Ux_free_count, Uy_free_count = around_free()
            for t in range(int(t_max)):
                #print(t / t_max)
                #  入射波
                if t < int(len(wave4)):
                    T1[0, sy_l:sy_r] = wave4[t]

                if t >= int(len(wave4)):
                    Uy[0, sy_l:sy_r] = 0
                    Ux[0, sy_l:sy_r] = 0
                    T1[0, 0:ny] = 0
                    T5[0, 0:ny] = 0

                #  応力の境界条件
                T5[0:nx, 0] = 0
                T5[0:nx, ny] = 0
                T3[0, 0:ny] = 0
                T1[nx, 0:ny] = 0
                T3[nx, 0:ny] = 0
                T1[0, 0] = 0
                T3[0, 0] = 0
                T5[0, 0] = 0
                #  横粒子速度の境界条件
                Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
                Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
                Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])
                #  きず部分の応力境界条件
                for i in range(num_f):
                    if (i + 1) * mn_p >= ny:
                        break
                    #  きず部分ゆえに消すとこ
                    T5[nx - mn_d:nx, mn_nf + i * mn_p:(i + 1) * mn_p] = 0
                    T1[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0
                    T3[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0

                #  横粒子速度の境界条件
                Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
                Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
                Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])

                #  応力の更新
                T1[1:nx, 0:ny] = T1[1:nx, 0:ny] - dtx * ((c11 * (Ux[1:nx, 0:ny] - Ux[0:nx - 1, 0:ny]))
                                                        + (c13 * (Uy[1:nx, 1:ny + 1] - Uy[1:nx, 0:ny])))

                T3[1:nx, 0:ny] = T3[1:nx, 0:ny] - dtx * ((c13 * (Ux[1:nx, 0:ny] - Ux[0:nx - 1, 0:ny]))
                                                        + (c11 * (Uy[1:nx, 1:ny + 1] - Uy[1:nx, 0:ny])))

                T5[0:nx, 1:ny] = T5[0:nx, 1:ny] - dtx * c55 * (Ux[0:nx, 1:ny] - Ux[0:nx, 0:ny - 1]
                                                            + Uy[1:nx + 1, 1:ny] - Uy[0:nx, 1:ny])
                #  きず部分の応力境界条件
                for i in range(num_f):
                    if (i + 1) * mn_p >= ny:
                        break
                    #  きず部分ゆえに消すとこ
                    T5[nx - mn_d:nx, mn_nf + i * mn_p:(i + 1) * mn_p] = 0
                    T1[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0
                    T3[nx - mn_d:nx + 1, mn_nf + i * mn_p:(i + 1) * mn_p - 1] = 0

                #  横粒子速度の境界条件
                Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
                Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
                Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])
                #  粒子速度の更新
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
                if t > 0:
                    wave[t] = cp.mean(T1[1, sy_l:sy_r])
                

            wave = cp.asnumpy(wave)

            #  """

            #  """
            print(f"f_pitch = {f_pitch}")
            print(f"f_depth = {f_depth}")
            name = f"C:\\Users\\manat\\project2\\data_all\\cupy_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_mesh{mesh_length}.csv"
            np.savetxt(name, wave, delimiter=',')

            # この時点でメモリ解放
            del T1, T3, T5, Ux, Uy, wave
            clear_memory()
            
        except Exception as e:
            print(f"Error occurred: {e}")
            clear_memory()
            continue

            # """
