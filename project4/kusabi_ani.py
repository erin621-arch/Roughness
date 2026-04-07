import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================== 調整パラメータ ==================

# 出力先
output_dir = r"C:/Users/cs16/Roughness/project4/tmp_output"  # 研究室PC
# output_dir = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder/tmp_output"   # 自宅PC

# ★くさび形 のパラメータ
f_pitch = 1.25e-3   # ピッチ p [m]
f_depth = 0.20e-3   # 深さ d [m]

# ★階段の高さ（メッシュ数）
step_size = 1

# ★アニメーション保存用
save_interval = 10        # 何ステップごとに1フレーム保存するか
global_downsample = 6     # 全体図の間引き率

# ★拡大表示の設定 
zoom_width_mm = 3.0
zoom_height_mm = 0.5

# ===================================================

def unique_path(path: str) -> str:
    """同名が存在したら _v2, _v3... を付けて上書き回避"""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 2
    while True:
        cand = f"{root}_v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def make_base_name_kusabi(f_pitch_m: float, f_depth_m: float, step: int) -> str:
    pitch_code = int(round(f_pitch_m * 1e5))
    depth_code = int(round(f_depth_m * 1e5))
    return f"kusabi_T1T3_pitch{pitch_code}_depth{depth_code}_step{int(step)}"

# ---------------- 基本パラメータ ----------------
x_length = 0.02   # [m]
y_length = 0.04   # [m]
mesh_length = 1.0e-5  # メッシュサイズ [m]

nx = int(x_length / mesh_length)
ny = int(y_length / mesh_length)

dx = x_length / nx
dy = y_length / ny

rho = 7840
E = 206 * 1e9
G = 80 * 1e9
V = 0.27

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)
f = 4.7e6  # 周波数
T = 1 / f
n = T / dt

# ---------------- 拡大領域ROI（Zoom） ----------------
roi_w_idx = int((zoom_width_mm * 1e-3) / dy)
y_center = ny // 2
y_start = max(0, y_center - roi_w_idx // 2)
y_end = min(ny, y_center + roi_w_idx // 2)

roi_h_idx = int((zoom_height_mm * 1e-3) / dx)
x_start = max(0, nx - roi_h_idx)
x_end = nx

print(f"ROI Indices: X[{x_start}:{x_end}], Y[{y_start}:{y_end}]")

# imshow extent（mm）
extent_global = [0, y_length * 1000, x_length * 1000, 0]
zoom_z_min = y_start * dy * 1000
zoom_z_max = y_end * dy * 1000
zoom_x_min = x_start * dx * 1000
zoom_x_max = x_end * dx * 1000
extent_zoom = [zoom_z_min, zoom_z_max, zoom_x_max, zoom_x_min]


# ---------------- くさび形マスク生成 ----------------
def isfree_kusabi(nx, ny, f_pitch, f_depth, mesh_length, step_size):
    # 1:固体 / 0:空洞
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree  = np.ones((nx, ny + 1))

    mn_p = int(round(f_pitch / mesh_length))
    mn_d = int(round(f_depth / mesh_length))

    # 外枠
    T13_isfree[0, 0:ny]  = 0
    T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0]   = 0
    T5_isfree[0:nx, ny]  = 0

    num_f = int(np.ceil(ny / mn_p))

    for i in range(num_f):
        y_s = i * mn_p
        y_e = min((i + 1) * mn_p, ny)

        for y in range(y_s, y_e):
            local_y = y - y_s
            ideal_depth = mn_d * (1.0 - (local_y) / mn_p)

            current_depth = (int(ideal_depth) // step_size) * step_size
            if current_depth <= 0:
                continue

            cut_top = max(0, nx - current_depth)
            T13_isfree[cut_top: nx + 1, y] = 0
            T5_isfree[cut_top: nx, y] = 0
            T5_isfree[cut_top: nx, y+1] = 0

    return T13_isfree, T5_isfree


# ---------------- 自由境界近傍の設定 ----------------
def around_free(T13_isfree, T5_isfree):
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

    for i in range(nx):
        for j in range(ny):
            if T13_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0: Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0: Ux_free_count[i, j] += 1

    '''
    for i in range(nx + 1):
        for j in range(ny + 1):
            if j == 0 or j == ny or i == 0 or i == nx:
                Uy_free_count[i, j] += 1
            elif 0 < i < nx and 0 < j < ny:
                if T13_isfree[i, j - 1] == 0: Uy_free_count[i, j] += 1
                if T13_isfree[i, j] == 0:     Uy_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0:  Uy_free_count[i, j] += 1
                if T5_isfree[i, j] == 0:      Uy_free_count[i, j] += 1
    '''
    
    # Uz 周囲カウント
    for i in range(nx + 1):
        for j in range(ny + 1):
            # 変更点1: i == nx をこの除外条件から消す
            if j == 0 or j == ny or i == 0:
                Uy_free_count[i, j] += 1  # 元のロジックに従い加算（完全に消すなら +=4 推奨）

            # 変更点2: i <= nx (底面を含む) まで範囲を広げる
            elif 0 < i <= nx and 0 < j < ny:
                if T13_isfree[i, j - 1] == 0: Uy_free_count[i, j] += 1
                if T13_isfree[i, j] == 0:     Uy_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0:  Uy_free_count[i, j] += 1
                
                # 変更点3: 下側の判定 (T5)
                if i < nx:
                    # 通常の内部：T5配列の中身をチェック
                    if T5_isfree[i, j] == 0:      Uy_free_count[i, j] += 1
                else:
                    # i == nx (底面) の場合：
                    # 下側は配列外（空気/Free）なので、無条件でカウントを加算
                    Uy_free_count[i, j] += 1

    return Ux_free_count, Uy_free_count


# ---------------- 入射波形 ----------------
wn = 2.5
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3

# ---------------- 計測設定 ----------------
sy = int(ny / 2)
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)

# くさび用は少し長めに計算
t_max = 1.3 * x_length / cl / dt


# =======================================================================
#               Viewer Class (Normalized, Titles, Scale Fixed)
# =======================================================================
class IntegratedViewer:
    # ★変更: mask_matrix を追加
    def __init__(self, root, g_t1, g_t3, z_t1, z_t3, time_step, output_dir,
                 extent_global, extent_zoom, roi_rect_mm, base_name, mask_matrix=None):
        self.root = root
        
        # --- データの正規化処理 (Normalize) ---
        max_t1 = np.max(np.abs(g_t1))
        if max_t1 > 1e-12:
            print(f"Normalizing T1 by max value: {max_t1:.4e}")
            self.g_t1 = g_t1 / max_t1
            self.z_t1 = z_t1 / max_t1
        else:
            self.g_t1 = g_t1
            self.z_t1 = z_t1

        max_t3 = np.max(np.abs(g_t3))
        if max_t3 > 1e-12:
            print(f"Normalizing T3 by max value: {max_t3:.4e}")
            self.g_t3 = g_t3 / max_t3
            self.z_t3 = z_t3 / max_t3
        else:
            self.g_t3 = g_t3
            self.z_t3 = z_t3
        
        self.time_step = time_step
        self.output_dir = output_dir
        self.base_name = base_name
        self.extent_global = extent_global
        self.extent_zoom = extent_zoom
        self.roi_rect_mm = roi_rect_mm
        self.mask_matrix = mask_matrix  # ★保存

        os.makedirs(self.output_dir, exist_ok=True)
        self.n_frames = self.g_t1.shape[0]
        self.play_running = False
        self.play_job = None

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Kusabi Simulation Viewer")
        self.root.geometry("1000x800")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(9, 7), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        self.ax_g1 = self.fig.add_subplot(gs[0, 0])
        self.ax_g3 = self.fig.add_subplot(gs[0, 1])
        self.ax_z1 = self.fig.add_subplot(gs[1, 0])
        self.ax_z3 = self.fig.add_subplot(gs[1, 1])

        vmin, vmax = -1.0, 1.0

        # Global T1
        self.im_g1 = self.ax_g1.imshow(
            self.g_t1[0], cmap="viridis", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="nearest", extent=self.extent_global
        )
        self.ax_g1.set_title("Global T1")
        self.ax_g1.set_ylabel("Depth X (mm)")
        self.add_roi_rect(self.ax_g1)
        self.draw_shape_outline(self.ax_g1, is_zoom=False) # ★輪郭線描画

        # Global T3
        self.im_g3 = self.ax_g3.imshow(
            self.g_t3[0], cmap="viridis", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="nearest", extent=self.extent_global
        )
        self.ax_g3.set_title("Global T3")
        self.add_roi_rect(self.ax_g3)
        self.draw_shape_outline(self.ax_g3, is_zoom=False) # ★輪郭線描画

        # Zoom T1
        self.im_z1 = self.ax_z1.imshow(
            self.z_t1[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="bilinear", extent=self.extent_zoom
        )
        self.ax_z1.set_title("Zoom T1")
        self.ax_z1.set_xlabel("Width Z (mm)"); self.ax_z1.set_ylabel("Depth X (mm)")
        self.draw_shape_outline(self.ax_z1, is_zoom=True) # ★輪郭線描画

        # Zoom T3
        self.im_z3 = self.ax_z3.imshow(
            self.z_t3[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="bilinear", extent=self.extent_zoom
        )
        self.ax_z3.set_title("Zoom T3")
        self.ax_z3.set_xlabel("Width Z (mm)")
        self.draw_shape_outline(self.ax_z3, is_zoom=True) # ★輪郭線描画

        self.fig.colorbar(self.im_z3, ax=[self.ax_g1, self.ax_g3, self.ax_z1, self.ax_z3], 
                          shrink=0.9, aspect=30, label="(Pa)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Control Panel ---
        control_frame = ttk.Frame(main_frame, relief=tk.RAISED, borderwidth=1)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)

        slider_box = ttk.Frame(control_frame)
        slider_box.pack(fill=tk.X, padx=10, pady=5)
        self.time_label = ttk.Label(slider_box, text="Time: 0.00 µs", width=20)
        self.time_label.pack(side=tk.LEFT)
        self.time_slider = ttk.Scale(slider_box, from_=0, to=self.n_frames - 1, orient=tk.HORIZONTAL, command=self.on_slider)
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        btn_box = ttk.Frame(control_frame)
        btn_box.pack(pady=5)
        ttk.Button(btn_box, text="<< Start", command=self.rewind_start).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_box, text="< Prev", command=self.step_back).pack(side=tk.LEFT, padx=5)
        self.btn_play = ttk.Button(btn_box, text="Play >", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_box, text="Next >", command=self.step_forward).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(btn_box, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=30.0)
        ttk.Scale(btn_box, from_=1, to=60, variable=self.speed_var, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)
        
        # --- 保存ボタン群 ---
        ttk.Button(btn_box, text="Save MP4", command=self.save_mp4).pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(btn_box, text="Save Data (.npz)", command=self.save_data).pack(side=tk.LEFT, padx=5)

        self.root.after(500, self.toggle_play)

    def draw_shape_outline(self, ax, is_zoom=False):
        """
        中身を塗らず、ブロックの「輪郭線（境界線）」だけをカクカク描画する
        （hlines, vlinesを使用）
        """
        if self.mask_matrix is None:
            return

        # --- 1. 描画するデータの準備 ---
        if is_zoom:
            # 拡大領域
            # 配列のスライス
            mask = self.mask_matrix[x_start:x_end, y_start:y_end]
            extent = self.extent_zoom
        else:
            # 全体表示用
            mask = self.mask_matrix[0:nx, :]
            extent = self.extent_global

        # マスクが空っぽなら何もしない
        if np.all(mask == 0) or np.all(mask == 1):
            return

        # --- 2. 座標計算の準備 ---
        # extent = [z_min, z_max, x_max, x_min] (origin='upper'の場合、y軸は上から下へ)
        z_min, z_max, x_bot, x_top = extent
        
        rows, cols = mask.shape
        
        # 1ピクセルあたりのサイズ
        dz = (z_max - z_min) / cols  # 横方向
        dx = (x_bot - x_top) / rows  # 縦方向

        # --- 3. 境界線の検出と描画 ---
        
        # 線の色と太さ
        line_color = 'black'
        line_width = 1.0 if is_zoom else 0.5

        # --- 縦線 (Vertical Lines) の描画 ---
        # 横方向に隣り合う値の差分をとる
        diff_h = np.diff(mask, axis=1)
        r_idx, c_idx = np.where(diff_h != 0)
        
        if len(r_idx) > 0:
            # x座標 (横軸)
            x_pos = z_min + (c_idx + 1) * dz
            # y座標 (縦軸)
            y_min = x_top + r_idx * dx
            y_max = x_top + (r_idx + 1) * dx
            
            ax.vlines(x_pos, y_min, y_max, colors=line_color, linewidths=line_width)

        # --- 横線 (Horizontal Lines) の描画 ---
        # 縦方向に隣り合う値の差分をとる
        diff_v = np.diff(mask, axis=0)
        r_idx, c_idx = np.where(diff_v != 0)
        
        if len(r_idx) > 0:
            # y座標 (縦軸)
            y_pos = x_top + (r_idx + 1) * dx
            # x座標 (横軸)
            x_min_line = z_min + c_idx * dz
            x_max_line = z_min + (c_idx + 1) * dz
            
            ax.hlines(y_pos, x_min_line, x_max_line, colors=line_color, linewidths=line_width)

    def add_roi_rect(self, ax):
        rz, rx, rw, rh = self.roi_rect_mm
        rect = Rectangle((rz, rx), rw, rh, linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    def update_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        # Update all 4 images
        self.im_g1.set_array(self.g_t1[frame_idx])
        self.im_g3.set_array(self.g_t3[frame_idx])
        self.im_z1.set_array(self.z_t1[frame_idx])
        self.im_z3.set_array(self.z_t3[frame_idx])

        current_t = frame_idx * self.time_step
        self.time_label.config(text=f"Time: {current_t*1e6:.2f} µs")
        self.canvas.draw_idle()

    def on_slider(self, val):
        self.update_frame(float(val))

    def toggle_play(self):
        if self.play_running:
            self.play_running = False
            self.btn_play.config(text="Play >")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None
        else:
            self.play_running = True
            self.btn_play.config(text="Pause ||")
            self.animate_loop()

    def animate_loop(self):
        if not self.play_running: return
        current_frame = int(self.time_slider.get())
        next_frame = current_frame + 1
        if next_frame >= self.n_frames: next_frame = 0
        self.time_slider.set(next_frame)
        self.update_frame(next_frame)
        delay = int(1000 / max(1e-9, self.speed_var.get()))
        self.play_job = self.root.after(delay, self.animate_loop)

    def rewind_start(self):
        self.play_running = False; self.btn_play.config(text="Play >")
        self.time_slider.set(0); self.update_frame(0)

    def step_back(self):
        self.play_running = False; self.btn_play.config(text="Play >")
        curr = self.time_slider.get(); self.time_slider.set(curr - 1); self.update_frame(curr - 1)

    def step_forward(self):
        self.play_running = False; self.btn_play.config(text="Play >")
        curr = self.time_slider.get(); self.time_slider.set(curr + 1); self.update_frame(curr + 1)

    def save_mp4(self):
        if self.play_running: self.toggle_play()
        fps = int(np.clip(self.speed_var.get(), 1, 60))
        mp4_path = unique_path(os.path.join(self.output_dir, f"{self.base_name}.mp4"))
        gif_path = unique_path(os.path.join(self.output_dir, f"{self.base_name}.gif"))

        def _update(i):
            i = int(i)
            self.im_g1.set_array(self.g_t1[i])
            self.im_g3.set_array(self.g_t3[i])
            self.im_z1.set_array(self.z_t1[i])
            self.im_z3.set_array(self.z_t3[i])
            current_t = i * self.time_step
            self.time_label.config(text=f"Time: {current_t*1e6:.2f} µs")
            return [self.im_g1, self.im_g3, self.im_z1, self.im_z3]

        ani = animation.FuncAnimation(self.fig, _update, frames=self.n_frames, interval=1000/fps, blit=False)
        try:
            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                ani.save(mp4_path, writer=writer, dpi=100)
                messagebox.showinfo("Saved", f"Saved MP4:\n{mp4_path}")
            else:
                writer = animation.PillowWriter(fps=fps)
                ani.save(gif_path, writer=writer, dpi=100)
                messagebox.showwarning("Saved (GIF)", f"ffmpeg missing, saved GIF:\n{gif_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed:\n{e}")

    # ★変更: mask_matrix も保存
    def save_data(self):
        if self.play_running: self.toggle_play()
        
        filename = unique_path(os.path.join(self.output_dir, f"{self.base_name}_data.npz"))
        
        try:
            print("Saving data... please wait.")
            np.savez_compressed(
                filename,
                g_t1=self.g_t1,
                g_t3=self.g_t3,
                z_t1=self.z_t1,
                z_t3=self.z_t3,
                time_step=self.time_step,
                extent_global=self.extent_global,
                extent_zoom=self.extent_zoom,
                roi_rect_mm=self.roi_rect_mm,
                base_name=self.base_name,
                mask_matrix=self.mask_matrix # ★追加
            )
            messagebox.showinfo("Saved", f"Data saved successfully:\n{filename}")
            print(f"Saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data:\n{e}")
            traceback.print_exc()

# =======================================================================
#               Main Simulation
# =======================================================================
def main():
    base_name = make_base_name_kusabi(f_pitch, f_depth, step_size)

    print(f"Pitch(p) = {f_pitch*1000} mm")
    print(f"Depth(d) = {f_depth*1000} mm")
    print(f"Step Size = {step_size} mesh(es)")
    print(f"save_interval = {save_interval}, global_downsample = {global_downsample}")
    print(f"Output dir: {output_dir}")
    print(f"Base name: {base_name}")

    os.makedirs(output_dir, exist_ok=True)

    # 配列確保
    T1 = cp.zeros((nx + 1, ny), dtype=cp.float32)
    T3 = cp.zeros((nx + 1, ny), dtype=cp.float32)
    T5 = cp.zeros((nx, ny + 1), dtype=cp.float32)
    Ux = cp.zeros((nx, ny), dtype=cp.float32)
    Uy = cp.zeros((nx + 1, ny + 1), dtype=cp.float32)

    dtx = dt / dx
    dty = dt / dy

    # マスク生成
    T13_isfree_np, T5_isfree_np = isfree_kusabi(nx, ny, f_pitch, f_depth, mesh_length, step_size)
    Ux_free_count_np, Uy_free_count_np = around_free(T13_isfree_np, T5_isfree_np)

    T13_isfree = cp.asarray(T13_isfree_np)
    T5_isfree = cp.asarray(T5_isfree_np)
    Ux_free_count = cp.asarray(Ux_free_count_np)
    Uy_free_count = cp.asarray(Uy_free_count_np)

    # フレーム保存用リスト
    g_frames_t1 = []
    g_frames_t3 = []
    z_frames_t1 = []
    z_frames_t3 = []

    start_time = time.time()
    print("Running Simulation...")

    for t in range(int(t_max)):
        if t % 500 == 0:
            print(f"{t}/{int(t_max)} ({t / t_max:.1%})")

        # 境界条件 (反射壁)
        T5[0:nx, 0] = 0; T5[0:nx, ny] = 0
        T3[0, 0:ny] = 0; T3[nx, 0:ny] = 0
        T1[nx, 0:ny] = 0
        T1[0, 0] = 0; T3[0, 0] = 0; T5[0, 0] = 0

        # Uy境界 (Kusabi特有の処理)
        Uy[1:nx, 0]  -= (4 / rho) * dtx * T3[1:nx, 0]
        Uy[1:nx, ny] -= (4 / rho) * dtx * (-T3[1:nx, ny - 1])
        Uy[nx, 1:ny] -= (4 / rho) * dtx * (-T5[nx - 1, 1:ny])

        # 応力更新
        T1[1:nx, :] -= dtx * (c11 * (Ux[1:nx, :] - Ux[0:nx-1, :]) + c13 * (Uy[1:nx, 1:] - Uy[1:nx, :-1]))
        T3[1:nx, :] -= dtx * (c13 * (Ux[1:nx, :] - Ux[0:nx-1, :]) + c11 * (Uy[1:nx, 1:] - Uy[1:nx, :-1]))
        T5[:, 1:ny] -= dtx * c55 * (Ux[:, 1:] - Ux[:, :-1] + Uy[1:, 1:ny] - Uy[:-1, 1:ny])

        # くさび内部の応力=0
        T1[T13_isfree == 0] = 0.0
        T3[T13_isfree == 0] = 0.0
        T5[T5_isfree[0:nx, :] == 0] = 0.0

        # 音源
        if t < int(len(wave4)):
            T1[0, sy_l:sy_r] = wave4[t]
        else:
            Uy[0, sy_l:sy_r] = 0
            Ux[0, sy_l:sy_r] = 0
            T1[0, 0:ny] = 0
            T5[0, 0:ny] = 0

        # 速度更新
        Ux[0:nx, 0:ny] = cp.where(
            Ux_free_count < 4,
            Ux - (4 / rho / (4 - Ux_free_count)) * dtx * (
                T1[1:nx+1, :] - T1[0:nx, :] + T5[:, 1:ny+1] - T5[:, 0:ny]
            ),
            0
        )
        Uy[1:nx, 1:ny] = cp.where(
            Uy_free_count[1:nx, 1:ny] < 4,
            Uy[1:nx, 1:ny] - (4 / rho / (4 - Uy_free_count[1:nx, 1:ny])) * dty * (
                T3[1:nx, 1:ny] - T3[1:nx, :-1] + T5[1:nx, 1:ny] - T5[:-1, 1:ny]
            ),
            0
        )

        # フレーム保存
        if t % save_interval == 0:
            # T1
            full1 = cp.asnumpy(T1[0:nx, :])
            g1 = full1[::global_downsample, ::global_downsample].astype(np.float32, copy=False)
            z1 = full1[x_start:x_end, y_start:y_end].astype(np.float32, copy=False)
            g_frames_t1.append(g1)
            z_frames_t1.append(z1)

            # T3
            full3 = cp.asnumpy(T3[0:nx, :])
            g3 = full3[::global_downsample, ::global_downsample].astype(np.float32, copy=False)
            z3 = full3[x_start:x_end, y_start:y_end].astype(np.float32, copy=False)
            g_frames_t3.append(g3)
            z_frames_t3.append(z3)

        if t % 1000 == 0:
            cp.cuda.Device().synchronize()

    print(f"Done. Time: {time.time() - start_time:.2f} s")

    # ---------------- Viewer起動 ----------------
    if len(g_frames_t1) == 0:
        print("No frames captured.")
        return

    g_data1 = np.array(g_frames_t1)
    g_data3 = np.array(g_frames_t3)
    z_data1 = np.array(z_frames_t1)
    z_data3 = np.array(z_frames_t3)

    roi_rect_mm = (
        zoom_z_min,
        zoom_x_min,
        (zoom_z_max - zoom_z_min),
        (zoom_x_max - zoom_x_min)
    )

    try:
        root = tk.Tk()
        _ = IntegratedViewer(
            root=root,
            g_t1=g_data1, g_t3=g_data3,
            z_t1=z_data1, z_t3=z_data3,
            time_step=dt * save_interval,
            output_dir=output_dir,
            extent_global=extent_global,
            extent_zoom=extent_zoom,
            roi_rect_mm=roi_rect_mm,
            base_name=base_name,
            mask_matrix=T13_isfree_np # ★変更: マスクを渡す
        )
        root.mainloop()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()