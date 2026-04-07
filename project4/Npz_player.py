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


class NpzPlayer:
    """
    NPZプレイヤー。以下の2フォーマットを自動判別して対応する。

    [single モード]  kusabi_ani_slide.py が出力するフォーマット
        必須キー: g_t1, time_step, extent_global, probe_rect_mm, base_name
        任意キー: mask_matrix
        表示: 1画面 (Global T1) + 探触子マーク(黄緑)

    [quad モード]  kusabi_ani.py / hanen_ani.py が出力するフォーマット
        必須キー: g_t1, g_t3, z_t1, z_t3, time_step,
                  extent_global, extent_zoom, roi_rect_mm, base_name
        任意キー: mask_matrix
        表示: 4画面 (Global T1/T3, Zoom T1/T3) + ROI矩形(赤)
    """

    def __init__(self, root, npz_path: str):
        self.root = root

        print(f"Loading: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        # ---- フォーマット自動判別 ----
        if "probe_rect_mm" in data:
            self.mode = "single"
        elif "roi_rect_mm" in data:
            self.mode = "quad"
        else:
            raise KeyError(
                "NPZフォーマット不明: 'probe_rect_mm' も 'roi_rect_mm' も見つかりません。"
            )

        # ---- 共通フィールド ----
        self.g_t1          = data["g_t1"]               # (n_frames, H, W)
        self.time_step     = float(data["time_step"])
        self.extent_global = data["extent_global"].tolist()
        self.base_name     = str(data["base_name"])
        self.mask_matrix   = data["mask_matrix"] if "mask_matrix" in data else None
        self.output_dir    = os.path.dirname(npz_path)
        self.n_frames      = self.g_t1.shape[0]

        # ---- モード別フィールド ----
        if self.mode == "single":
            self.probe_rect_mm = data["probe_rect_mm"].tolist()
        else:
            self.g_t3        = data["g_t3"]
            self.z_t1        = data["z_t1"]
            self.z_t3        = data["z_t3"]
            self.extent_zoom = data["extent_zoom"].tolist()
            self.roi_rect_mm = data["roi_rect_mm"].tolist()

        print(f"  mode       : {self.mode}")
        print(f"  frames     : {self.n_frames}")
        print(f"  frame shape: {self.g_t1.shape[1:]}")
        print(f"  time_step  : {self.time_step*1e9:.4f} ns")
        print(f"  base_name  : {self.base_name}")

        self.play_running = False
        self.play_job     = None

        self._setup_ui()

    # ------------------------------------------------------------------
    #  UI 構築
    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.root.title(f"NPZ Player [{self.mode}]  —  {self.base_name}")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True)

        if self.mode == "single":
            self._build_single_view(graph_frame)
        else:
            self._build_quad_view(graph_frame)

        # ---- 共通コントロールパネル ----
        ctrl = ttk.Frame(main_frame, relief=tk.RAISED, borderwidth=1)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)

        slider_box = ttk.Frame(ctrl)
        slider_box.pack(fill=tk.X, padx=10, pady=5)
        self.time_label = ttk.Label(slider_box, text="Time: 0.00 µs", width=20)
        self.time_label.pack(side=tk.LEFT)
        self.time_slider = ttk.Scale(
            slider_box, from_=0, to=self.n_frames - 1,
            orient=tk.HORIZONTAL, command=self._on_slider
        )
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
    #  Single ビュー (kusabi_ani_slide フォーマット)
    # ------------------------------------------------------------------
    def _build_single_view(self, parent):
        self.root.geometry("800x700")
        self.fig = plt.Figure(figsize=(8, 6), dpi=100, layout="constrained")
        ax = self.fig.add_subplot(111)

        self.im_g1 = ax.imshow(
            self.g_t1[0], cmap="viridis", vmin=-1.0, vmax=1.0,
            aspect="auto", interpolation="nearest",
            extent=self.extent_global, zorder=1
        )
        ax.set_title("Global T1")
        ax.set_xlabel("Width Z (mm)")
        ax.set_ylabel("Depth X (mm)")

        if self.mask_matrix is not None:
            self._draw_white_fill(ax, self.mask_matrix, self.extent_global)
            self._draw_shape_outline(ax, self.mask_matrix, self.extent_global)

        pz, px, pw, ph = self.probe_rect_mm
        ax.add_patch(Rectangle((pz, px), pw, ph,
                                linewidth=3, edgecolor="lawngreen",
                                facecolor="none", zorder=4))

        self.fig.colorbar(self.im_g1, ax=ax, shrink=0.9, aspect=30, label="(norm.)")
        self._attach_canvas(parent)

    # ------------------------------------------------------------------
    #  Quad ビュー (kusabi_ani / hanen_ani フォーマット)
    # ------------------------------------------------------------------
    def _build_quad_view(self, parent):
        self.root.geometry("1000x800")
        self.fig = plt.Figure(figsize=(9, 7), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(2, 2)

        ax_g1 = self.fig.add_subplot(gs[0, 0])
        ax_g3 = self.fig.add_subplot(gs[0, 1])
        ax_z1 = self.fig.add_subplot(gs[1, 0])
        ax_z3 = self.fig.add_subplot(gs[1, 1])

        vmin, vmax = -1.0, 1.0

        self.im_g1 = ax_g1.imshow(
            self.g_t1[0], cmap="viridis", vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest", extent=self.extent_global, zorder=1
        )
        ax_g1.set_title("Global T1")
        ax_g1.set_ylabel("Depth X (mm)")

        self.im_g3 = ax_g3.imshow(
            self.g_t3[0], cmap="viridis", vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest", extent=self.extent_global, zorder=1
        )
        ax_g3.set_title("Global T3")

        self.im_z1 = ax_z1.imshow(
            self.z_t1[0], cmap="RdBu_r", vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="bilinear", extent=self.extent_zoom, zorder=1
        )
        ax_z1.set_title("Zoom T1")
        ax_z1.set_xlabel("Width Z (mm)")
        ax_z1.set_ylabel("Depth X (mm)")

        self.im_z3 = ax_z3.imshow(
            self.z_t3[0], cmap="RdBu_r", vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="bilinear", extent=self.extent_zoom, zorder=1
        )
        ax_z3.set_title("Zoom T3")
        ax_z3.set_xlabel("Width Z (mm)")

        # ROI矩形 (Global ビューに赤い枠)
        rz, rx, rw, rh = self.roi_rect_mm
        for ax in (ax_g1, ax_g3):
            ax.add_patch(Rectangle((rz, rx), rw, rh,
                                    linewidth=1.5, edgecolor="red",
                                    facecolor="none", zorder=4))

        # マスク描画
        if self.mask_matrix is not None:
            mask_global = self.mask_matrix[:self.mask_matrix.shape[0]-1, :]
            mask_zoom   = self._crop_mask_to_zoom(mask_global)
            for ax in (ax_g1, ax_g3):
                self._draw_white_fill(ax, mask_global, self.extent_global)
                self._draw_shape_outline(ax, mask_global, self.extent_global)
            if mask_zoom is not None:
                for ax in (ax_z1, ax_z3):
                    self._draw_shape_outline(ax, mask_zoom, self.extent_zoom, linewidth=1.0)

        self.fig.colorbar(self.im_z3,
                          ax=[ax_g1, ax_g3, ax_z1, ax_z3],
                          shrink=0.9, aspect=30, label="(norm.)")
        self._attach_canvas(parent)

    def _attach_canvas(self, parent):
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    #  マスク描画ユーティリティ
    # ------------------------------------------------------------------
    def _draw_white_fill(self, ax, mask, extent):
        """空洞部分(mask==0)を白で塗りつぶす"""
        nx_ = mask.shape[0]
        rgba = np.zeros((nx_, mask.shape[1], 4), dtype=np.float32)
        rgba[mask == 0] = [1.0, 1.0, 1.0, 1.0]
        ax.imshow(rgba, extent=extent, aspect="auto",
                  interpolation="nearest", zorder=2)

    def _draw_shape_outline(self, ax, mask, extent, linewidth=0.5):
        """形状の境界線だけを黒線で描く"""
        if np.all(mask == 0) or np.all(mask == 1):
            return
        z_min, z_max, x_bot, x_top = extent
        rows, cols = mask.shape
        dz = (z_max - z_min) / cols
        dx = (x_bot - x_top) / rows

        diff_h = np.diff(mask, axis=1)
        r_idx, c_idx = np.where(diff_h != 0)
        if len(r_idx) > 0:
            ax.vlines(z_min + (c_idx + 1) * dz,
                      x_top + r_idx * dx, x_top + (r_idx + 1) * dx,
                      colors='black', linewidths=linewidth, zorder=3)

        diff_v = np.diff(mask, axis=0)
        r_idx, c_idx = np.where(diff_v != 0)
        if len(r_idx) > 0:
            ax.hlines(x_top + (r_idx + 1) * dx,
                      z_min + c_idx * dz, z_min + (c_idx + 1) * dz,
                      colors='black', linewidths=linewidth, zorder=3)

    def _crop_mask_to_zoom(self, mask_global):
        """extent_zoom に対応するマスクの部分行列を返す"""
        if mask_global is None:
            return None
        z_min, z_max, x_bot, x_top = self.extent_global
        zz_min, zz_max, xx_bot, xx_top = self.extent_zoom
        rows, cols = mask_global.shape
        dz = (z_max - z_min) / cols
        dx = (x_bot - x_top) / rows
        c0 = int(round((zz_min - z_min) / dz))
        c1 = int(round((zz_max - z_min) / dz))
        r0 = int(round((xx_top - x_top) / dx))
        r1 = int(round((xx_bot - x_top) / dx))
        c0, c1 = max(0, c0), min(cols, c1)
        r0, r1 = max(0, r0), min(rows, r1)
        if r1 <= r0 or c1 <= c0:
            return None
        return mask_global[r0:r1, c0:c1]

    # ------------------------------------------------------------------
    #  フレーム更新
    # ------------------------------------------------------------------
    def _update_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        self.im_g1.set_array(self.g_t1[frame_idx])
        if self.mode == "quad":
            self.im_g3.set_array(self.g_t3[frame_idx])
            self.im_z1.set_array(self.z_t1[frame_idx])
            self.im_z3.set_array(self.z_t3[frame_idx])
        current_t = frame_idx * self.time_step
        self.time_label.config(text=f"Time: {current_t*1e6:.2f} µs")
        self.canvas.draw_idle()

    def _on_slider(self, val):
        self._update_frame(float(val))

    # ------------------------------------------------------------------
    #  再生コントロール
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  MP4保存
    # ------------------------------------------------------------------
    def _save_mp4(self):
        if self.play_running:
            self._toggle_play()
        fps = int(np.clip(self.speed_var.get(), 1, 60))
        mp4_path = unique_path(os.path.join(self.output_dir, f"{self.base_name}.mp4"))
        gif_path = unique_path(os.path.join(self.output_dir, f"{self.base_name}.gif"))

        if self.mode == "single":
            def _update_ani(i):
                self.im_g1.set_array(self.g_t1[i])
                self.time_label.config(text=f"Time: {i*self.time_step*1e6:.2f} µs")
                return [self.im_g1]
        else:
            def _update_ani(i):
                self.im_g1.set_array(self.g_t1[i])
                self.im_g3.set_array(self.g_t3[i])
                self.im_z1.set_array(self.z_t1[i])
                self.im_z3.set_array(self.z_t3[i])
                self.time_label.config(text=f"Time: {i*self.time_step*1e6:.2f} µs")
                return [self.im_g1, self.im_g3, self.im_z1, self.im_z3]

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
    NpzPlayer(root, npz_path)
    root.mainloop()


if __name__ == "__main__":
    main()
