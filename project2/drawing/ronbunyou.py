import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
# project2ディレクトリを直接指定
sys.path.insert(0, r'C:\Users\manat\project2')
import os
import sourse_new
import matplotlib.patches as patches

plt.rcParams["font.size"] = 13



sim_ref = -np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth0.csv")
exp_ref = sourse_new.import_data(0, 0)[0]

sim_rough = -np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth20.csv")
exp_rough = sourse_new.import_data(125, 20)[0]

sim_ref = sourse_new.kiritori2(sourse_new.interpolate_sim_one(sim_ref[9000:]), sourse_new.left, sourse_new.right)
exp_ref = sourse_new.kiritori2(exp_ref, sourse_new.left, sourse_new.right)

sim_rough = sourse_new.kiritori2(sourse_new.interpolate_sim_one(sim_rough[9000:]), sourse_new.left, sourse_new.right)
exp_rough = sourse_new.kiritori2(exp_rough, sourse_new.left, sourse_new.right)

yf_sim_ref, freq = sourse_new.make_fftdata(sim_ref, sourse_new.exp_dt)
yf_exp_ref, freq = sourse_new.make_fftdata(exp_ref, sourse_new.exp_dt)

yf_sim_rough, freq = sourse_new.make_fftdata(sim_rough, sourse_new.exp_dt)
yf_exp_rough, freq = sourse_new.make_fftdata(exp_rough, sourse_new.exp_dt)



# print(len(sim_ref))
# print(len(exp_ref))


# fig, ax = plt.subplots(figsize=(6, 6))
# # step2 y軸の作成
# twin1 = ax.twinx()
# t = np.arange(0, sourse_new.exp_dt*len(sim_ref), sourse_new.exp_dt)

# p1, = ax.plot(t, exp_ref, label='Smooth')
# p2, = twin1.plot(t, exp_rough, label='Rough', color="orange")
# # 囲みたい範囲を指定（時間軸の範囲）
# highlight_start = 1550*sourse_new.exp_dt  # 開始時間
# highlight_end = t[-1]

# # 四角形を描画（twin1の軸に合わせる）
# rect = patches.Rectangle((highlight_start, twin1.get_ylim()[0]), 
#                         highlight_end - highlight_start, 
#                         twin1.get_ylim()[1] - twin1.get_ylim()[0],
#                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
# twin1.add_patch(rect)
# # 四角形の上に文字を追加
# # 左辺を上まで延長する縦線
# ax.axvline(x=highlight_start, ymin=0, ymax=1, color='red', linestyle='--', linewidth=2)

# text_x = (highlight_start + highlight_end) / 2  # 四角形の中央
# text_y = twin1.get_ylim()[1] + 0.2  # 四角形の上端より少し上
# twin1.text(text_x, text_y, 'Tail wave', 
#            ha='center', va='bottom', fontsize=14, color='red', weight='bold')
# fig.tight_layout()
# ax.set_ylim(-3.0, 1.5)
# twin1.set_ylim(-0.7, 3)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Voltage [V]')
# twin1.set_ylabel('Voltage [V]', rotation=270)
# # step4 凡例の追加
# ax.legend(handles=[p1, p2])
# plt.tight_layout()


# plt.savefig(r"C:\Users\manat\project2\tmp_output\exp_waves.eps", format="eps", dpi=300)
# plt.show()

# fig, ax = plt.subplots(figsize=(6, 6))
# # step2 y軸の作成
# twin1 = ax.twinx()
# t = np.arange(0, sourse_new.exp_dt*len(sim_ref), sourse_new.exp_dt)

# p1, = ax.plot(t, sim_ref, label='Smooth')
# p2, = twin1.plot(t, sim_rough, label='Rough', color="orange")

# # 囲みたい範囲を指定（時間軸の範囲）
# highlight_start = 1550*sourse_new.exp_dt  # 開始時間
# highlight_end = t[-1]  # 終了時間

# # 四角形を描画（twin1の軸に合わせる）
# rect = patches.Rectangle((highlight_start, twin1.get_ylim()[0]), 
#                         highlight_end - highlight_start, 
#                         twin1.get_ylim()[1] - twin1.get_ylim()[0],
#                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
# twin1.add_patch(rect)
# # 四角形の上に文字を追加
# # 左辺を上まで延長する縦線
# ax.axvline(x=highlight_start, ymin=0, ymax=1, color='red', linestyle='--', linewidth=2)

# # 四角形の上に文字を追加
# text_x = (highlight_start + highlight_end) / 2  # 四角形の中央
# text_y = twin1.get_ylim()[1] + 0.2  # 四角形の上端より少し上
# twin1.text(text_x, text_y, 'Tail wave', 
#            ha='center', va='bottom', fontsize=14, color='red', weight='bold')
# ax.set_ylim(-5.0, 2.2)
# twin1.set_ylim(-1.5, 5.0)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Pressure [Pa]')
# twin1.set_ylabel('Pressure [Pa]', rotation=270)
# # step4 凡例の追加
# ax.legend(handles=[p1, p2])
# plt.tight_layout()
# plt.savefig(r"C:\Users\manat\project2\tmp_output\sim_waves.eps", format="eps", dpi=300)
# plt.show()

# plt.rcParams["font.size"] = 14
# plt.figure()
# plt.plot(freq, yf_exp_ref, label="Smooth surface")
# plt.plot(freq, yf_exp_rough, label="Rough surface")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.xlim(0, 10e6)

# plt.savefig(r"C:\Users\manat\project2\tmp_output\Figure6a.eps", format="eps", dpi=300)
# plt.show()


# sourse_new.select_pitch_and_image(125, [5, 10, 14, 20])
# sourse_new.select_pitch_and_image(150, [4, 10, 15, 20])
# sourse_new.select_pitch_and_image(175, [4, 10, 15, 21])
# sourse_new.select_pitch_and_image(200, [4, 10, 17, 20])
# sourse_new.select_pitch_and_image_exp(125, [5, 10, 14, 20])
# sourse_new.select_pitch_and_image_exp(150, [4, 10, 15, 20])
# sourse_new.select_pitch_and_image_exp(175, [4, 10, 15, 21])
# sourse_new.select_pitch_and_image_exp(200, [4, 10, 17, 20])

# filename = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T1\\T1_series_middle_morerange.npy"
# filename2 = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T3\\T3_series_middle_morerange.npy"
# data_kabe_T1 = np.load(filename)
# data_yuka_T3 = np.load(filename2)

filename = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T1\\T1_series_middle_moremorerange_200_moredepth_tyoizurasi.npy"
filename2 = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T3\\T3_series_middle_moremorerange_200_moredepth_tyoizurasi.npy"
data_kabe_T1 = np.load(filename)
data_yuka_T3 = np.load(filename2)

z1 = 49
z2 = 224
x1 = 30
x2 = 40

# fig = plt.figure()
# im = plt.imshow(-data_kabe_T1[x1:x2, z1, 1214-sourse_new.left:1214+sourse_new.right], extent=(0, sourse_new.right+sourse_new.left, 0, 20),interpolation='nearest', cmap="jet", aspect='auto')
# plt.xlabel("Timestep")
# plt.ylabel("Amplitude")
# cbar = plt.colorbar(im)
# ticks = cbar.get_ticks() 
# ticklabels = [ticklabel.get_text() for ticklabel in cbar.ax.get_yticklabels()]
# ticklabels[-1] += ' [Pa]'
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(ticklabels)
# plt.show()

# fig = plt.figure()
# im = plt.imshow(-data_yuka_T3[x2-2, z1:z2, 1214-sourse_new.left:1214+sourse_new.right], extent=(0, sourse_new.right+sourse_new.left, 0, 20),interpolation='nearest', cmap="jet", aspect='auto')
# plt.xlabel("Timestep")
# plt.ylabel("Amplitude")
# cbar = plt.colorbar(im)
# ticks = cbar.get_ticks() 
# ticklabels = [ticklabel.get_text() for ticklabel in cbar.ax.get_yticklabels()]
# ticklabels[-1] += ' [Pa]'
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(ticklabels)
# plt.show()

dt_ns = 0.71  # ns per timestep
time_range = sourse_new.right + sourse_new.left
# プローブ発射を0とした実時間軸
start_time_step = 4000  # 描画開始時のタイムステップ
time_axis_us = [(start_time_step + i) * dt_ns / 1000 for i in range(time_range)]

fig = plt.figure()
im = plt.imshow(-data_kabe_T1[x1-10:x2, z1, 1214-sourse_new.left:1214+sourse_new.right], 
                extent=(time_axis_us[0], time_axis_us[-1], 0, 0.20),
                interpolation='nearest', cmap="bwr", aspect='auto')
plt.xlabel("Time [μs]")
plt.ylabel(r"x-axis position [mm]")

# 現在の目盛りを取得して、最大値を追加
ax = plt.gca()
current_yticks = list(ax.get_yticks())
if 0.10 not in current_yticks:
    current_yticks.append(0.10)
    ax.set_yticks(sorted(current_yticks))


plt.text(0.02, 0.1, '(a)', transform=plt.gca().transAxes,
    fontsize=25, weight='bold', ha='left', va='top')
cbar = plt.colorbar(im)
cbar.set_label('[Pa]', rotation=0, labelpad=10)
plt.savefig(r"C:\Users\manat\project2\drawing\jasa\Figure100a.png", format="eps", dpi=300)
plt.show()

plt.rcParams["font.size"] = 13
fig = plt.figure()
im = plt.imshow(-data_yuka_T3[x2-2, z1:z2, 1214-sourse_new.left:1214+sourse_new.right], 
                extent=(time_axis_us[0], time_axis_us[-1], 0, 1.75),
                interpolation='nearest', cmap="bwr", aspect='auto')
plt.xlabel("Time [μs]")
plt.ylabel(r"z-axis position [mm]")

# 2つ目のプロットでも同様に
ax = plt.gca()
current_yticks = list(ax.get_yticks())
if 1.75 not in current_yticks:
    current_yticks = [tick for tick in current_yticks if abs(tick - 1.75) > 0.01]
    current_yticks.append(1.75)
    ax.set_yticks(sorted(current_yticks))

ax.set_ylim(0, 1.75)  # y軸の範囲を強制的に0-1.75に固定
plt.text(0.02, 0.1, '(b)', transform=plt.gca().transAxes,
    fontsize=25, weight='bold', ha='left', va='top')
cbar = plt.colorbar(im)
cbar.set_label('[Pa]', rotation=0, labelpad=10)
plt.savefig(r"C:\Users\manat\project2\drawing\jasa\Figure100b.png", format="png", dpi=300)
plt.show()





# wave200_10 = sourse_new.import_data(200, 10)[0]

# y = sourse_new.kiritori2(sourse_new.interpolate_sim_one_inverse(wave200_10), sourse_new.left, sourse_new.right)

# wave200_10_sim = -np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch200_depth10.csv")
# y2 = sourse_new.kiritori2(wave200_10_sim, sourse_new.left, sourse_new.right)


# # サンプルデータ
# x = np.arange(len(y))


# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# # 上：imshow
# im = ax1.imshow(-data_kabe_T1[x1:x2, z1, 1214-sourse_new.left+100:1214+sourse_new.right+100], extent=(0, sourse_new.right+sourse_new.left, 0, 10), aspect='auto', cmap='jet')
# ax1.set_ylabel("x axis")

# # 下：plot
# ax3.plot(x, y)
# ax3.set_ylabel('Voltage [V]')
# ax3.set_xlabel('Timestep')

# # 下：plot
# ax2.plot(x, y2)
# ax2.set_ylabel('Pressure [Pa]')

# plt.tight_layout()
# plt.show()


wave125_20 = sourse_new.import_data(125, 20)[0]
wave200_20 = sourse_new.import_data(200, 20)[0]
yf200_20, freq = sourse_new.make_fftdata_main_tale(sourse_new.kiritori2
                                                   (sourse_new.interpolate_sim_one(-np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch200_depth20.csv")[9000:]), sourse_new.left, sourse_new.right), sourse_new.exp_dt, 1500)
yf125_20, freq = sourse_new.make_fftdata_main_tale(sourse_new.kiritori2
                                                   (sourse_new.interpolate_sim_one(-np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth20.csv")[9000:]), sourse_new.left, sourse_new.right), sourse_new.exp_dt, 1550)

filename_125_20 = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T3\\T3_series_middle_moremorerange_125_moredepth.npy"
filename_200_20 = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T3\\T3_series_middle_moremorerange_moredepth.npy"
data1 = np.load(filename_200_20)


z1 = 49
z2 = 224
z3 = 200
z4 = 300
x2 = 69

# middle_200 = 135

# yf_surface_125, freq_surface = sourse_new.make_fftdata(sourse_new.kiritori2(sourse_new.interpolate_sim_one(data1[-1, middle_200, :]), sourse_new.left, sourse_new.right), sourse_new.exp_dt)

# plt.rcParams["font.size"] = 14
# plt.plot(freq[0], yf200_20[2]/yf200_20[2].max(), label="Tail wave")
# plt.plot(freq_surface, yf_surface_125/yf_surface_125.max(), label="Surface wave")

# plt.xlim(0, 10000000)
# plt.ylim(0, 1.2)
# plt.ylabel("Min-Max normalized Amplitude")
# plt.xlabel("Frequency [Hz]")
# plt.tight_layout()
# plt.legend(loc='upper center', ncol=2)
# plt.savefig(r"C:\Users\manat\project2\tmp_output\Figure10b.eps", format="eps", dpi=300)
# plt.show()


# wave125_20 = sourse_new.interpolate_sim_one(-np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch125_depth20.csv")[9000:])
# wave200_20 = sourse_new.interpolate_sim_one(-np.loadtxt(r"C:\Users\manat\project2\data_all\cupy_pitch200_depth20.csv")[9000:])
yf200_20, freq = sourse_new.make_fftdata_main_tale(sourse_new.kiritori2(wave200_20, sourse_new.left, sourse_new.right), sourse_new.exp_dt, 1900)
yf125_20, freq = sourse_new.make_fftdata_main_tale(sourse_new.kiritori2(wave125_20, sourse_new.left, sourse_new.right), sourse_new.exp_dt, 1900)

# filename_125_20 = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T3\\T3_series_middle_moremorerange_125_moredepth.npy"
# filename_200_20 = f"C:\\Users\\manat\\project2\\surface_wave_2d\\T3\\T3_series_middle_moremorerange_moredepth.npy"
data1 = np.load(filename_125_20)
data2 = np.load(filename_200_20)

z1 = 49
z2 = 224
z3 = 49
z4 = 225
x2 = 69



yf_surface_200, freq_surface = sourse_new.make_fftdata(sourse_new.interpolate_sim_one(sourse_new.kiritori2(data2[-1, int((z3+z4)/2), :], sourse_new.left, sourse_new.right+1)), sourse_new.exp_dt)

# plt.rcParams["font.size"] = 14
# plt.plot(freq[2], yf200_20[2]/yf200_20[2].max(), label="Sim. Tail wave")
# plt.plot(freq_surface, yf_surface_200/yf_surface_200.max(), label="Sim. Surface wave")

# plt.xlim(0, 10000000)
# plt.ylim(0, 1.2)
# plt.ylabel("Min-Max normalized Amplitude")
# plt.xlabel("Frequency [Hz]")
# plt.tight_layout()
# plt.legend(loc='upper center', ncol=2)
# plt.savefig(r"C:\Users\manat\project2\tmp_output\Figure10c.eps", format="eps", dpi=300)
# plt.show()

# t = np.arange(0, len(sim_ref[:2000])*sourse_new.dt, sourse_new.dt)
# plt.figure()
# plt.plot(t, sim_ref[:2000])
# plt.xlabel("Time [s]")
# plt.ylabel("Pressure [Pa]")
# plt.tight_layout()
# plt.savefig(r"C:\Users\manat\project2\tmp_output\Figure11a.jpg", format="jpg", dpi=300)
# plt.show()