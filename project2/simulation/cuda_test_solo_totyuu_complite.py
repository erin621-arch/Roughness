##########################################################################
#  シミュレーションを傷のバリエーション全部でやるやつ

import numpy as np
import cupy as cp

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
f = 4.7e6  # frequency
T = 1 / f  # period
lam = cl / f  # lambda
k = 1 / lam  # wave number
n = T / dt  # 波が離散点上で何点か

start_point = 1799
end_point = 1975



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

print(len(wave4))
print(cl)

# 音源の位置
sy = 1887+200
sx = 0
# 探触子の直径 m
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)


t_max = 4 * x_length / cl / dt  # 1往復ちょいの時間

#  =====================================================================================

wave = np.zeros(int(t_max))
# 傷の幅 m
f_width = 0.25e-3  # m
# 傷の間隔 m
f_pitch = 2.00e-3
# 傷の深さ m
f_depth = 0.10e-3 # m

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

series_range = 14000
T1 = cp.zeros((nx + 1, ny), dtype=float)
T3 = cp.zeros((nx + 1, ny), dtype=float)
T5 = cp.zeros((nx, ny + 1), dtype=float)
Ux = cp.zeros((nx, ny), dtype=float)
Uy = cp.zeros((nx + 1, ny + 1), dtype=float)
T1_series_yoko = np.zeros((int(np.round(f_pitch/mesh_length - f_width/mesh_length))+1, 14000), dtype=float)
T1_series_tate = np.zeros((int (f_depth + 0.00015 / mesh_length), 14000), dtype=float)
T1_series_tate_1799 = np.zeros((int (f_depth + 0.00015 / mesh_length), 14000), dtype=float)
T3_series_yoko = np.zeros((int(np.round(f_pitch/mesh_length - f_width/mesh_length))+1, 14000), dtype=float)
T3_series_tate = np.zeros((int (f_depth + 0.00015 / mesh_length), 14000), dtype=float)
T3_series_tate_1799 = np.zeros((int (f_depth + 0.00015 / mesh_length), 14000), dtype=float)


dtx = dt / dx
dty = dt / dy

T13_isfree, T5_isfree = isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length)
Ux_free_count, Uy_free_count = around_free()
for t in range(int(t_max)):
    print(t / t_max)
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
        if t > 4000 and t < 18000:
            T1_series_tate[:, t - 4001] = cp.asnumpy(T1[nx - int (f_depth + 0.00015 / mesh_length):nx,
                                start_point])
            T1_series_tate_1799[:, t - 4001] = cp.asnumpy(T1[nx - int ( f_depth + 0.00015/ mesh_length):nx,
                                end_point-1])
            T1_series_yoko[:, t - 4001] = cp.asnumpy(T1[nx - int (f_depth + 0.00015 / mesh_length):nx,
                                start_point:end_point])[-1, :]
            T3_series_tate[:, t - 4001] = cp.asnumpy(T3[nx - int (f_depth + 0.00015 / mesh_length):nx,
                                start_point])
            T3_series_tate_1799[:, t - 4001] = cp.asnumpy(T3[nx - int (f_depth + 0.00015 / mesh_length):nx,
                                end_point-1])
            T3_series_yoko[:, t - 4001] = cp.asnumpy(T3[nx - int (f_depth + 0.00015 / mesh_length):nx,
                                start_point:end_point])[-1, :]        
        
   

wave = cp.asnumpy(wave)

#  """

#  """
for i in range(14000):
        filename = f"C:\\Users\\manat\\project2\\surface_wave_data\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_T3_tate_t{i}_{start_point}.csv"
        np.savetxt(filename, T3_series_tate[:, i], delimiter=',')
        filename = f"C:\\Users\\manat\\project2\\surface_wave_data\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_T3_tate_{start_point}_t{i}_{start_point}.csv"
        np.savetxt(filename, T3_series_tate_1799[:, i], delimiter=',')
        filename = f"C:\\Users\\manat\\project2\\surface_wave_data\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_T3_yoko_t{i}_{start_point}.csv"
        np.savetxt(filename, T3_series_yoko[:, i], delimiter=',')
        filename = f"C:\\Users\\manat\\project2\\surface_wave_data\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_T1_tate_t{i}_{start_point}.csv"
        np.savetxt(filename, T1_series_tate[:, i], delimiter=',')
        filename = f"C:\\Users\\manat\\project2\\surface_wave_data\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_T1_tate_{start_point}_t{i}_{start_point}.csv"
        np.savetxt(filename, T1_series_tate_1799[:, i], delimiter=',')
        filename = f"C:\\Users\\manat\\project2\\surface_wave_data\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}_T1_yoko_t{i}_{start_point}.csv"
        np.savetxt(filename, T1_series_yoko[:, i], delimiter=',')
print(f"f_pitch = {f_pitch}")
print(f"f_depth = {f_depth}")
name = f"tmp_output\\mannaka2_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}.csv"
np.savetxt(name, wave, delimiter=',')

# """
