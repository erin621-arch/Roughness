import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. パスの設定とデータの読み込み
# ==========================================
# ※実際の .npz ファイルのパスに合わせて変更してください
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"
npz_filename = "ushape_surface_map_center_pitch200_depth20.npz" 
npz_path = os.path.join(output_dir, npz_filename)

print(f"データを読み込んでいます: {npz_path}")
data = np.load(npz_path)

# データの抽出
T1_corners_r = data['T1_corners_r'] # 形状: (計測点数, 時間ステップ数)
corner_x_r = data['corner_x_r']
corner_z_r = data['corner_z_r']
t_rec_start = data['t_rec_start']
t_rec_len = data['t_rec_len']
dt = data['dt']

# ==========================================
# 2. 時間軸の作成とインデックスの指定
# ==========================================
# 時間軸をマイクロ秒（μs）単位で作成
time_axis = (np.arange(t_rec_len) + t_rec_start) * dt * 1e6

# 比較したい計測点のインデックスを指定
# 0: 1番目の点（最も深い底付近 X:1980, Z:2017）
# 3: 4番目の点（斜面の中腹 X:1983, Z:2021）
idx_deepest = 0
idx_4th = 3

# ==========================================
# 3. グラフのプロット
# ==========================================
plt.figure(figsize=(10, 5))

# 最も深い点の波形を青色でプロット
label_deepest = f"Deepest Point 1 (X:{corner_x_r[idx_deepest]}, Z:{corner_z_r[idx_deepest]})"
plt.plot(time_axis, T1_corners_r[idx_deepest, :], label=label_deepest, color='blue', linewidth=1.5)

# 4番目の点の波形をオレンジ色でプロット
label_4th = f"Point 4 (X:{corner_x_r[idx_4th]}, Z:{corner_z_r[idx_4th]})"
plt.plot(time_axis, T1_corners_r[idx_4th, :], label=label_4th, color='orange', linewidth=1.5)

# グラフの装飾
plt.title("Comparison of T1 Waveforms at U-shape Corners", fontsize=14)
plt.xlabel("Time [μs]", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# ==========================================
# 4. 画像の保存と表示
# ==========================================
# 保存したい場合は以下のコメントアウトを外してください
# save_path = os.path.join(output_dir, "T1_waveform_comparison.png")
# plt.savefig(save_path, dpi=300)
# print(f"グラフを保存しました: {save_path}")

plt.show()
