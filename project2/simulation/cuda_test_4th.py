##########################################################################
#  境界まで4次精度FDTD改良版

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

# 音源の位置
sy = int(ny / 2)
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
f_pitch = 1.50e-3
# 傷の深さ m
f_depth = 0.03e-3  # m

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

def calc_derivative_4th_x(field, dx):
    """x方向の4次精度微分（境界も含む）"""
    derivative = cp.zeros_like(field)
    
    # 内部領域：中央差分4次精度
    derivative[2:-2, :] = (-field[4:, :] + 8*field[3:-1, :] - 8*field[1:-3, :] + field[:-4, :]) / (12*dx)
    
    # 左境界：前方差分4次精度
    derivative[0, :] = (-25*field[0, :] + 48*field[1, :] - 36*field[2, :] + 16*field[3, :] - 3*field[4, :]) / (12*dx)
    derivative[1, :] = (-3*field[0, :] - 10*field[1, :] + 18*field[2, :] - 6*field[3, :] + field[4, :]) / (12*dx)
    
    # 右境界：後方差分4次精度
    derivative[-2, :] = (-field[-5, :] + 6*field[-4, :] - 18*field[-3, :] + 10*field[-2, :] + 3*field[-1, :]) / (12*dx)
    derivative[-1, :] = (3*field[-5, :] - 16*field[-4, :] + 36*field[-3, :] - 48*field[-2, :] + 25*field[-1, :]) / (12*dx)
    
    return derivative

def calc_derivative_4th_y(field, dy):
    """y方向の4次精度微分（境界も含む）"""
    derivative = cp.zeros_like(field)
    
    # 内部領域：中央差分4次精度
    derivative[:, 2:-2] = (-field[:, 4:] + 8*field[:, 3:-1] - 8*field[:, 1:-3] + field[:, :-4]) / (12*dy)
    
    # 下境界：前方差分4次精度
    derivative[:, 0] = (-25*field[:, 0] + 48*field[:, 1] - 36*field[:, 2] + 16*field[:, 3] - 3*field[:, 4]) / (12*dy)
    derivative[:, 1] = (-3*field[:, 0] - 10*field[:, 1] + 18*field[:, 2] - 6*field[:, 3] + field[:, 4]) / (12*dy)
    
    # 上境界：後方差分4次精度
    derivative[:, -2] = (-field[:, -5] + 6*field[:, -4] - 18*field[:, -3] + 10*field[:, -2] + 3*field[:, -1]) / (12*dy)
    derivative[:, -1] = (3*field[:, -5] - 16*field[:, -4] + 36*field[:, -3] - 48*field[:, -2] + 25*field[:, -1]) / (12*dy)
    
    return derivative

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

    #  応力の更新 (全域4次精度)
    dUx_dx = calc_derivative_4th_x(Ux, dx)
    dUy_dy = calc_derivative_4th_y(Uy[1:nx+1, :], dy)
    
    T1[1:nx, 0:ny] = T1[1:nx, 0:ny] - dt * (c11 * dUx_dx[1:nx, 0:ny] + c13 * dUy_dy[0:nx-1, 0:ny])
    T3[1:nx, 0:ny] = T3[1:nx, 0:ny] - dt * (c13 * dUx_dx[1:nx, 0:ny] + c11 * dUy_dy[0:nx-1, 0:ny])

    # T5の更新 (全域4次精度)
    dUx_dy = calc_derivative_4th_y(Ux, dy)
    dUy_dx = calc_derivative_4th_x(Uy[:, 1:ny+1], dx)
    
    T5[0:nx, 1:ny] = T5[0:nx, 1:ny] - dt * c55 * (dUx_dy[0:nx, 1:ny] + dUy_dx[0:nx, 0:ny-1])

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

    #  粒子速度の更新 (全域4次精度)
    dT1_dx = calc_derivative_4th_x(T1, dx)
    dT5_dy = calc_derivative_4th_y(T5, dy)
    
    # 自由境界に接するノードと接しないノードを組み合わせて粒子速度を更新
    Ux[0:nx, 0:ny] = cp.where(cp.asarray(Ux_free_count[0:nx, 0:ny]) < 4,
                                Ux[0:nx, 0:ny] - (4 / rho / (4 - cp.asarray(Ux_free_count[0:nx, 0:ny]))) * dt
                                * (dT1_dx[0:nx, 0:ny] + dT5_dy[0:nx, 0:ny]), 0)

    # Uyの更新も全域4次精度で実装
    dT3_dy = calc_derivative_4th_y(T3, dy)
    dT5_dx = calc_derivative_4th_x(T5, dx)
    
    Uy[1:nx, 1:ny] = cp.where(cp.asarray(Uy_free_count[1:nx, 1:ny]) < 4,
                                Uy[1:nx, 1:ny] - (4 / rho / (4 - cp.asarray(Uy_free_count[1:nx, 1:ny]))) * dt
                                * (dT3_dy[1:nx, 1:ny] + dT5_dx[1:nx, 1:ny]), 0)

    cp.cuda.Device().synchronize()

wave = cp.asnumpy(wave)

print(f"f_pitch = {f_pitch}")
print(f"f_depth = {f_depth}")
name = f"tmp_output\\cupy_4th_boundary_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}.csv"
np.savetxt(name, wave, delimiter=',')