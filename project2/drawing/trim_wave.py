import numpy as np
import matplotlib.pyplot as plt

file_path1 = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series_middle_moremorerange_moredepth.npy"
file_path2 = r"C:\Users\manat\project2\surface_wave_2d\T3\T3_series_middle_moremorerange_moredepth.npy"

data1 = np.load(file_path1) + np.load(file_path2)

x = 10
z = 135
plt.plot(data1[x, z, :])
plt.show()