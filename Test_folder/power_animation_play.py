import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib import animation
import traceback

# =======================================================================
#               Universal Player (Multi-Window Supported)
# =======================================================================

def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 2
    while True:
        cand = f"{root}_v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

class IntegratedViewer:
    def __init__(self, window, g_t1, g_t3, z_t1, z_t3, time_step, output_dir, 
                 extent_global, extent_zoom, roi_rect_mm, base_name, main_app_root, mask_matrix=None):
        self.window = window        # このビューワーのウィンドウ (Toplevel)
        self.main_app_root = main_app_root # アプリ全体の管理ルート (tk.Tk)
        
        # データ格納
        self.g_t1 = g_t1
        self.z_t1 = z_t1
        self.g_t3 = g_t3
        self.z_t3 = z_t3
        
        self.time_step = time_step
        self.output_dir = output_dir
        self.base_name = base_name
        self.extent_global = extent_global
        self.extent_zoom = extent_zoom
        self.roi_rect_mm = roi_rect_mm
        self.mask_matrix = mask_matrix

        os.makedirs(self.output_dir, exist_ok=True)
        self.n_frames = self.g_t1.shape[0]
        self.play_running = False
        self.play_job = None

        self.setup_ui()

    def setup_ui(self):
        # ウィンドウを閉じたときの処理
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.window.title(f"Player: {self.base_name}")
        self.window.geometry("1000x800") 

        main_frame = ttk.Frame(self.window)
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

        # --- 描画 (Global T1) ---
        self.im_g1 = self.ax_g1.imshow(
            self.g_t1[0], cmap="viridis", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="nearest", extent=self.extent_global
        )
        self.ax_g1.set_title("Global T1")
        self.ax_g1.set_ylabel("Depth X (mm)")
        self.add_roi_rect(self.ax_g1)
        # 全体図はそのまま描画
        self.draw_kusabi_outline(self.ax_g1)

        # --- 描画 (Global T3) ---
        self.im_g3 = self.ax_g3.imshow(
            self.g_t3[0], cmap="viridis", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="nearest", extent=self.extent_global
        )
        self.ax_g3.set_title("Global T3")
        self.add_roi_rect(self.ax_g3)
        self.draw_kusabi_outline(self.ax_g3)

        # --- 描画 (Zoom T1) ---
        self.im_z1 = self.ax_z1.imshow(
            self.z_t1[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="bilinear", extent=self.extent_zoom
        )
        self.ax_z1.set_title("Zoom T1")
        self.ax_z1.set_xlabel("Width Z (mm)"); self.ax_z1.set_ylabel("Depth X (mm)")
        # ★ここが重要：Zoom図は描画後に範囲を強制的に再設定する
        self.draw_kusabi_outline(self.ax_z1, enforce_extent=self.extent_zoom)

        # --- 描画 (Zoom T3) ---
        self.im_z3 = self.ax_z3.imshow(
            self.z_t3[0], cmap="RdBu_r", vmin=vmin, vmax=vmax, 
            aspect="auto", interpolation="bilinear", extent=self.extent_zoom
        )
        self.ax_z3.set_title("Zoom T3")
        self.ax_z3.set_xlabel("Width Z (mm)")
        # ★ここが重要：Zoom図は描画後に範囲を強制的に再設定する
        self.draw_kusabi_outline(self.ax_z3, enforce_extent=self.extent_zoom)

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
        
        ttk.Button(btn_box, text="Save MP4", command=self.save_mp4).pack(side=tk.LEFT, padx=(20, 5))
        
        self.window.after(500, self.toggle_play)

    def add_roi_rect(self, ax):
        rz, rx, rw, rh = self.roi_rect_mm
        rect = Rectangle((rz, rx), rw, rh, linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    # ★修正: enforce_extent 引数を追加
    def draw_kusabi_outline(self, ax, enforce_extent=None):
        """くさび形状の境界線（等高線）を描画する"""
        if self.mask_matrix is None:
            return

        rows, cols = self.mask_matrix.shape
        mask_view = self.mask_matrix[0:rows-1, :]

        # 全体基準で描画（これによりAxisが自動拡大されてしまう）
        ax.contour(mask_view, levels=[0.5], colors='black', linewidths=0.8, 
                   origin='upper', extent=self.extent_global)
        
        # enforce_extent が指定されている場合（＝Zoom図）、範囲を強制的に戻す
        if enforce_extent is not None:
            # extent = [left, right, bottom, top]
            ax.set_xlim(enforce_extent[0], enforce_extent[1])
            ax.set_ylim(enforce_extent[2], enforce_extent[3])

    def update_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
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
                self.window.after_cancel(self.play_job)
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
        self.play_job = self.window.after(delay, self.animate_loop)

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

    def on_close(self):
        self.play_running = False
        if self.play_job:
            self.window.after_cancel(self.play_job)
        self.window.destroy()
        remaining_windows = [w for w in self.main_app_root.winfo_children() if isinstance(w, tk.Toplevel)]
        if not remaining_windows:
            self.main_app_root.destroy()

def load_and_play():
    main_root = tk.Tk()
    main_root.withdraw()
    
    file_paths = filedialog.askopenfilenames(
        title="Select Simulation Data (.npz)",
        filetypes=[("NumPy Zip", "*.npz")]
    )

    if not file_paths:
        main_root.destroy()
        return

    for file_path in file_paths:
        try:
            data = np.load(file_path)
            g_t1 = data['g_t1']
            g_t3 = data['g_t3']
            z_t1 = data['z_t1']
            z_t3 = data['z_t3']
            time_step = float(data['time_step'])
            extent_global = data['extent_global']
            extent_zoom = data['extent_zoom']
            roi_rect_mm = data['roi_rect_mm']
            base_name = str(data['base_name'])
            mask_matrix = data.get('mask_matrix') 
            
            output_dir = os.path.dirname(file_path)

            window = tk.Toplevel(main_root)
            _ = IntegratedViewer(
                window=window,
                g_t1=g_t1, g_t3=g_t3,
                z_t1=z_t1, z_t3=z_t3,
                time_step=time_step,
                output_dir=output_dir,
                extent_global=extent_global,
                extent_zoom=extent_zoom,
                roi_rect_mm=roi_rect_mm,
                base_name=f"Replay_{base_name}",
                main_app_root=main_root,
                mask_matrix=mask_matrix
            )
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load {os.path.basename(file_path)}:\n{e}")
            traceback.print_exc()

    main_root.mainloop()

if __name__ == "__main__":
    load_and_play()