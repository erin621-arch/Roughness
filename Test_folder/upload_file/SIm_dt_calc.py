#シミュレーションのdtの計算

import numpy as np

# ---------------- パラメータ設定 ----------------
x_length = 0.02       # x方向の長さ [m]
mesh_length = 1.0e-5  # メッシュ長 [m]

rho = 7840            # 密度 [kg/m^3]
E = 206 * 1e9         # ヤング率
V = 0.27              

# ---------------- 計算処理 ----------------
# メッシュサイズ dx の計算
nx = int(x_length / mesh_length)
dx = x_length / nx

# 縦波速度 cl (P波) の計算
cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))

# 時間刻み dt の計算
dt = dx / cl / np.sqrt(6)

# ---------------- 結果表示 ----------------
print(f"P波速度 (cl): {cl:.2f} m/s")
print(f"時間刻み (dt): {dt:.5e} s")
print(f"時間刻み (dt): {dt * 1e9:.5f} ns")