import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


class KusabiNpzPlayer:
    def __init__(self, root, npz_path: str):
        self.root = root

        print(f"Loading: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        self.g_t1        = data["g_t1"]          # (n_frames, nx_ds, ny_ds)
        self.time_step   = float(data["time_step"])
        self.extent_global = data["extent_global"].tolist()
        self.probe_rect_mm = data["probe_rect_mm"].tolist()
        self.base_name   = str(data["base_name"])
        self.mask_matrix = data["mask_matrix"] if "mask_matrix" in data else None

        self.output_dir  = os.path.dirname(npz_path)
        self.n_frames    = self.g_t1.shape[0]

        print(f"  frames     : {self.n_frames}")
        print(f"  frame shape: {self.g_t1.shape[1:]}")
        print(f"  time_step  : {self.time_step*1e9:.4f} ns")
        print(f"  base_name  : {self.base_name}")

        self.play_running = False
        self.play_job = None

        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.root.title(f"Kusabi NPZ Player  —  {self.base_name}")
        self.root.geometry("800x700")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.ax  = self.fig.add_subplot(111)

        self.im = self.ax.imshow(
            self.g_t1[0], cmap="viridis", vmin=-1.0, vmax=1.0,
            aspect="auto", interpolation="nearest",
            extent=self.extent_global, zorder=1
        )
        self.ax.set_title("Global T1")
        self.ax.set_xlabel("Width Z (mm)")
        self.ax.set_ylabel("Depth X (mm)")

        # きず部分を白く塗りつぶす
        if self.mask_matrix is not None:
            nx_full = self.mask_matrix.shape[0]
            mask = self.mask_matrix[0:nx_full, :]
            rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
            rgba_mask[mask == 0] = [1.0, 1.0, 1.0, 1.0]
            self.ax.imshow(rgba_mask, extent=self.extent_global,
                           aspect="auto", interpolation="nearest", zorder=2)
            self._draw_shape_outline(self.ax)

        # 探触子マーク
        pz, px, pw, ph = self.probe_rect_mm
        probe_patch = Rectangle((pz, px), pw, ph,
                                 linewidth=3, edgecolor="lawngreen",
                                 facecolor="none", zorder=4)
        self.ax.add_patch(probe_patch)

        self.fig.colorbar(self.im, ax=self.ax, shrink=0.9, aspect=30, label="(norm.)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- コントロールパネル ---
        ctrl = ttk.Frame(main_frame, relief=tk.RAISED, borderwidth=1)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)

        slider_box = ttk.Frame(ctrl)
        slider_box.pack(fill=tk.X, padx=10, pady=5)
        self.time_label = ttk.Label(slider_box, text="Time: 0.00 µs", width=20)
        self.time_label.pack(side=tk.LEFT)
        self.time_slider = ttk.Scale(slider_box, from_=0, to=self.n_frames - 1,
                                     orient=tk.HORIZONTAL, command=self._on_slider)
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        btn_box = ttk.Frame(ctrl)
        btn_box.pack(pady=5)
        ttk.Button(btn_box, text="<< Start",  command=self._rewind).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_box, text="< Prev",    command=self._step_back).pack(side=tk.LEFT, padx=5)
        self.btn_play = ttk.Button(btn_box, text="Play >", command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_box, text="Next >",    command=self._step_forward).pack(side=tk.LEFT, padx=5)

        ttk.Label(btn_box, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=30.0)
        ttk.Scale(btn_box, from_=1, to=60, variable=self.speed_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)

        ttk.Button(btn_box, text="Save MP4",
                   command=self._save_mp4).pack(side=tk.LEFT, padx=(20, 5))

        self.root.after(500, self._toggle_play)

    # ------------------------------------------------------------------
    def _draw_shape_outline(self, ax):
        if self.mask_matrix is None:
            return
        nx_full = self.mask_matrix.shape[0]
        mask    = self.mask_matrix[0:nx_full, :]
        extent  = self.extent_global

        if np.all(mask == 0) or np.all(mask == 1):
            return

        z_min, z_max, x_bot, x_top = extent
        rows, cols = mask.shape
        dz = (z_max - z_min) / cols
        dx = (x_bot - x_top) / rows

        diff_h = np.diff(mask, axis=1)
        r_idx, c_idx = np.where(diff_h != 0)
        if len(r_idx) > 0:
            x_pos = z_min + (c_idx + 1) * dz
            y_min = x_top + r_idx * dx
            y_max = x_top + (r_idx + 1) * dx
            ax.vlines(x_pos, y_min, y_max, colors='black', linewidths=0.5, zorder=3)

        diff_v = np.diff(mask, axis=0)
        r_idx, c_idx = np.where(diff_v != 0)
        if len(r_idx) > 0:
            y_pos     = x_top + (r_idx + 1) * dx
            x_min_ln  = z_min + c_idx * dz
            x_max_ln  = z_min + (c_idx + 1) * dz
            ax.hlines(y_pos, x_min_ln, x_max_ln, colors='black', linewidths=0.5, zorder=3)

    # ------------------------------------------------------------------
    def _update_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        self.im.set_array(self.g_t1[frame_idx])
        current_t = frame_idx * self.time_step
        self.time_label.config(text=f"Time: {current_t*1e6:.2f} µs")
        self.canvas.draw_idle()

    def _on_slider(self, val):
        self._update_frame(float(val))

    def _toggle_play(self):
        if self.play_running:
            self.play_running = False
            self.btn_play.config(text="Play >")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None
        else:
            self.play_running = True
            self.btn_play.config(text="Pause ||")
            self._animate_loop()

    def _animate_loop(self):
        if not self.play_running:
            return
        current = int(self.time_slider.get())
        next_f  = (current + 1) % self.n_frames
        self.time_slider.set(next_f)
        self._update_frame(next_f)
        delay = int(1000 / max(1e-9, self.speed_var.get()))
        self.play_job = self.root.after(delay, self._animate_loop)

    def _rewind(self):
        self.play_running = False
        self.btn_play.config(text="Play >")
        self.time_slider.set(0)
        self._update_frame(0)

    def _step_back(self):
        self.play_running = False
        self.btn_play.config(text="Play >")
        curr = self.time_slider.get()
        self.time_slider.set(curr - 1)
        self._update_frame(curr - 1)

    def _step_forward(self):
        self.play_running = False
        self.btn_play.config(text="Play >")
        curr = self.time_slider.get()
        self.time_slider.set(curr + 1)
        self._update_frame(curr + 1)

    def _save_mp4(self):
        if self.play_running:
            self._toggle_play()
        fps = int(np.clip(self.speed_var.get(), 1, 60))
        mp4_path = unique_path(os.path.join(self.output_dir, f"{self.base_name}.mp4"))
        gif_path = unique_path(os.path.join(self.output_dir, f"{self.base_name}.gif"))

        def _update_ani(i):
            self.im.set_array(self.g_t1[i])
            current_t = i * self.time_step
            self.time_label.config(text=f"Time: {current_t*1e6:.2f} µs")
            return [self.im]

        ani = animation.FuncAnimation(self.fig, _update_ani,
                                      frames=self.n_frames,
                                      interval=1000 / fps, blit=False)
        try:
            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                ani.save(mp4_path, writer=writer, dpi=100)
                messagebox.showinfo("Saved", f"Saved MP4:\n{mp4_path}")
            else:
                writer = animation.PillowWriter(fps=fps)
                ani.save(gif_path, writer=writer, dpi=100)
                messagebox.showwarning("Saved (GIF)",
                                       f"ffmpeg が見つからないため GIF で保存:\n{gif_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"保存に失敗:\n{e}")


# =======================================================================
#   エントリポイント
# =======================================================================
def main():
    # ファイル選択ダイアログ
    tmp = tk.Tk()
    tmp.withdraw()
    npz_path = filedialog.askopenfilename(
        title="NPZ ファイルを選択",
        initialdir=r"C:/Users/cs16/Roughness/project4/tmp_output",
        filetypes=[("NumPy compressed", "*.npz"), ("All files", "*.*")]
    )
    tmp.destroy()

    if not npz_path:
        print("ファイルが選択されませんでした。終了します。")
        return

    root = tk.Tk()
    KusabiNpzPlayer(root, npz_path)
    root.mainloop()


if __name__ == "__main__":
    main()
