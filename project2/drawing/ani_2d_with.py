import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import os
import traceback
import japanize_matplotlib




dt_01 = 7.1282897372711e-10
dt_05 = 3.562362796201232e-09

ratio = dt_05/dt_01


class DualTimeScaleViewer:
    def __init__(self, root, data_detail, data_overview, detail_time_step=1, overview_time_step=5):
        """
        異なる時間刻みを持つ2つのデータセットを同期表示するビューア
        
        Parameters:
        -----------
        root : tkinter.Tk
            tkinterのルートウィンドウ
        data_detail : numpy.ndarray
            詳細表示用の3次元データ (時間, 縦, 横)
        data_overview : numpy.ndarray
            概要表示用の3次元データ (時間, 縦, 横)
        detail_time_step : float/int
            詳細データの時間刻み（相対値）
        overview_time_step : float/int
            概要データの時間刻み（相対値）
        """
        self.root = root
        self.data_detail = data_detail
        self.data_overview = data_overview
        
        # 時間次元の取得
        self.n_frames_detail = data_detail.shape[0]
        self.n_frames_overview = data_overview.shape[0]
        
        # 時間刻みの比率を計算
        self.detail_time_step = detail_time_step
        self.overview_time_step = overview_time_step
        self.time_ratio = overview_time_step / detail_time_step
        
        # 総時間を計算（詳細データの時間単位で）
        self.total_time = (self.n_frames_detail - 1) * detail_time_step
        
        print(f"詳細データ: {self.n_frames_detail}フレーム, 時間刻み: {detail_time_step}")
        print(f"概要データ: {self.n_frames_overview}フレーム, 時間刻み: {overview_time_step}")
        print(f"時間刻み比率: {self.time_ratio}")
        print(f"総時間: {self.total_time}")
        
        # UIのセットアップ
        self.setup_ui()
        
    def setup_ui(self):
        """UIコンポーネントの初期化"""
        self.root.title("超音波シミュレーション可視化ツール - 時間同期表示")

        
        
        # ウィンドウサイズを設定
        window_width = 1200
        window_height = 800
        self.root.geometry(f"{window_width}x{window_height}")
        
        # メインフレームを作成
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # グラフ表示用フレーム
        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # matplotlibのfigureを作成（2つのサブプロット）
        self.fig = plt.Figure(figsize=(12, 10), dpi=100)
        
        # サブプロットを作成（縦に2つ配置、比率は3:1）
        gs = self.fig.add_gridspec(4, 1)
        self.ax_detail = self.fig.add_subplot(gs[0:3, 0])  # 上3/4を詳細表示に
        self.ax_overview = self.fig.add_subplot(gs[3, 0])  # 下1/4を概要表示に
        
        # 詳細データの縦横比を計算
        height_detail = self.data_detail.shape[1]
        width_detail = self.data_detail.shape[2]
        aspect_ratio_detail = width_detail / height_detail
        print(f"詳細データの縦横比: {aspect_ratio_detail:.2f}:1")
        
        # 概要データの縦横比を計算
        height_overview = self.data_overview.shape[1]
        width_overview = self.data_overview.shape[2]
        aspect_ratio_overview = width_overview / height_overview
        print(f"概要データの縦横比: {aspect_ratio_overview:.2f}:1")
        
        # データの値の範囲を計算
        vmin_detail = np.min(self.data_detail)
        vmax_detail = np.max(self.data_detail)
        vmin_overview = np.min(self.data_overview)
        vmax_overview = np.max(self.data_overview)
        
        # 詳細データの表示
        self.im_detail = self.ax_detail.imshow(
            self.data_detail[0], 
            cmap='viridis', 
            vmin=vmin_detail, 
            vmax=vmax_detail, 
            aspect='auto', 
            interpolation='bilinear'
        )
        
        # 概要データの表示
        self.im_overview = self.ax_overview.imshow(
            self.data_overview[0], 
            cmap='viridis', 
            vmin=vmin_overview, 
            vmax=vmax_overview, 
            aspect='auto', 
            interpolation='bilinear'
        )
        
    
        self.title_overview = self.ax_overview.set_title(f't = 0') # 概要プロット用のタイトルを設定
        self.title_detail = self.ax_detail.set_title(f't = 0')  # この行を追加

        # カラーバーを追加
        cbar_detail = self.fig.colorbar(self.im_detail, ax=self.ax_detail, orientation='vertical', pad=0.01)

        
        cbar_overview = self.fig.colorbar(self.im_overview, ax=self.ax_overview, orientation='vertical', pad=0.01)

        
        
        self.ax_detail.set_xlabel('Z')
        self.ax_detail.set_ylabel('X')
        self.ax_overview.set_xlabel('Z')
        self.ax_overview.set_ylabel('X')
        
        # 縦軸の目盛りを調整
        self.ax_detail.set_yticks(np.linspace(0, height_detail-1, min(height_detail, 5)).astype(int))
        self.ax_overview.set_yticks(np.linspace(0, height_overview-1, min(height_overview, 3)).astype(int))
        
        # 横軸の目盛りを調整
        self.ax_detail.set_xticks(np.linspace(0, width_detail-1, min(width_detail, 10)).astype(int))
        self.ax_overview.set_xticks(np.linspace(0, width_overview-1, min(width_overview, 10)).astype(int))
        
        # レイアウト調整
        self.fig.tight_layout()
        
        # matplotlibのfigureをtkinterに埋め込む
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ナビゲーションツールバーを追加
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()
        
        # 時間表示部分のフレーム
        time_frame = ttk.Frame(main_frame)
        time_frame.pack(fill=tk.X, pady=5)
        
        # 現在の時間表示
        self.time_label = ttk.Label(time_frame, text=f"t: 0.00 / {self.total_time:.2f}")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # 詳細フレーム表示
        self.detail_frame_label = ttk.Label(time_frame, text=f"frame: 0 / {self.n_frames_detail-1}")
        self.detail_frame_label.pack(side=tk.LEFT, padx=15)
        
        # 概要フレーム表示
        self.overview_frame_label = ttk.Label(time_frame, text=f"frame: 0 / {self.n_frames_overview-1}")
        self.overview_frame_label.pack(side=tk.LEFT, padx=15)
        
        # スライダー用フレーム
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=5)
        
        # 時間スライダー変更時の処理
        def update_from_slider(val):
            time_val = float(val)
            self.update_display(time_val)
        
        # スライダーは時間単位で操作（0から総時間まで）
        self.time_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=self.total_time,
            orient=tk.HORIZONTAL,
            command=update_from_slider,
            length=window_width-100
        )
        self.time_slider.pack(fill=tk.X, expand=True, padx=10)
        
        # 時間入力フィールド
        time_input_frame = ttk.Frame(main_frame)
        time_input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(time_input_frame, text="t:").pack(side=tk.LEFT, padx=5)
        
        # 現在の時間値を表示するエントリフィールド
        self.time_var = tk.StringVar(value="0.00")
        time_entry = ttk.Entry(time_input_frame, textvariable=self.time_var, width=10)
        time_entry.pack(side=tk.LEFT, padx=5)
        
        # エントリフィールド更新時の処理
        def entry_update(event=None):
            try:
                time_val = float(self.time_var.get())
                if 0 <= time_val <= self.total_time:
                    self.time_slider.set(time_val)
                    self.update_display(time_val)
            except:
                pass
        
        time_entry.bind('<Return>', entry_update)
        
        set_time_button = ttk.Button(time_input_frame, text="setting", command=entry_update)
        set_time_button.pack(side=tk.LEFT, padx=5)
        
        # フレーム直接入力部分
        ttk.Label(time_input_frame, text="frame:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.detail_frame_var = tk.StringVar(value="0")
        detail_frame_entry = ttk.Entry(time_input_frame, textvariable=self.detail_frame_var, width=8)
        detail_frame_entry.pack(side=tk.LEFT, padx=5)
        
        # 詳細フレーム更新時の処理
        def detail_frame_update(event=None):
            try:
                frame = int(self.detail_frame_var.get())
                if 0 <= frame < self.n_frames_detail:
                    # フレームから時間を計算
                    time_val = frame * self.detail_time_step
                    # スライダーと表示を更新
                    self.time_slider.set(time_val)
                    self.update_display(time_val)
            except:
                pass
        
        detail_frame_entry.bind('<Return>', detail_frame_update)
        
        set_detail_button = ttk.Button(time_input_frame, text="setting", command=detail_frame_update)
        set_detail_button.pack(side=tk.LEFT, padx=5)
        
        # 概要フレーム直接入力
        ttk.Label(time_input_frame, text="frame:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.overview_frame_var = tk.StringVar(value="0")
        overview_frame_entry = ttk.Entry(time_input_frame, textvariable=self.overview_frame_var, width=8)
        overview_frame_entry.pack(side=tk.LEFT, padx=5)
        
        # 概要フレーム更新時の処理
        def overview_frame_update(event=None):
            try:
                frame = int(self.overview_frame_var.get())
                if 0 <= frame < self.n_frames_overview:
                    # フレームから時間を計算
                    time_val = frame * self.overview_time_step
                    # スライダーと表示を更新
                    self.time_slider.set(time_val)
                    self.update_display(time_val)
            except:
                pass
        
        overview_frame_entry.bind('<Return>', overview_frame_update)
        
        set_overview_button = ttk.Button(time_input_frame, text="setting", command=overview_frame_update)
        set_overview_button.pack(side=tk.LEFT, padx=5)
        
        # 操作ボタン用フレーム
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # 再生/停止用の変数
        self.play_state = {"running": False, "job": None}
        
        # 再生ボタンのコールバック
        def play_animation():
            if self.play_state["running"]:
                # 停止
                if self.play_state["job"] is not None:
                    self.root.after_cancel(self.play_state["job"])
                    self.play_state["job"] = None
                self.play_state["running"] = False
                self.play_button.config(text="再生")
            else:
                # 再生
                self.play_state["running"] = True
                self.play_button.config(text="停止")
                self.animate_next_frame()
        
        # フレーム毎のアニメーション
        def step_animation(step_size):
            current_time = float(self.time_slider.get())
            new_time = current_time + step_size * self.detail_time_step
            
            # 範囲チェック
            if new_time < 0:
                new_time = 0
            elif new_time > self.total_time:
                new_time = self.total_time
                
            # 更新
            self.time_slider.set(new_time)
            self.update_display(new_time)
            
        
        # 再生/停止ボタン
        self.play_button = ttk.Button(button_frame, text="再生", command=play_animation)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # 前へボタン
        prev_button = ttk.Button(button_frame, text="前へ", command=lambda: step_animation(-1))
        prev_button.pack(side=tk.LEFT, padx=5)
        
        # 次へボタン
        next_button = ttk.Button(button_frame, text="次へ", command=lambda: step_animation(1))
        next_button.pack(side=tk.LEFT, padx=5)
        
        # リセットボタン
        def reset_animation():
            self.time_slider.set(0)
            self.update_display(0)
        
        reset_button = ttk.Button(button_frame, text="リセット", command=reset_animation)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # ループ再生チェックボックス
        self.loop_var = tk.BooleanVar(value=True)
        loop_check = ttk.Checkbutton(button_frame, text="ループ再生", variable=self.loop_var)
        loop_check.pack(side=tk.LEFT, padx=10)
        
        # 速度調整用のラベルとスライダー
        ttk.Label(button_frame, text="速度:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.speed_var = tk.DoubleVar(value=40000.0)  # デフォルト400fps
        speed_slider = ttk.Scale(
            button_frame,
            from_=400,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            length=150
        )
        speed_slider.pack(side=tk.LEFT)
        
        # 現在のFPS表示
        self.fps_label = ttk.Label(button_frame, text="15 fps")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        # FPS更新
        def update_fps(event=None):
            fps = self.speed_var.get()
            self.fps_label.config(text=f"{int(fps)} fps")
        
        speed_slider.bind("<Motion>", update_fps)
        
        # キーボードショートカット
        def key_handler(event):
            if event.keysym == 'Right':
                step_animation(1)
            elif event.keysym == 'Left':
                step_animation(-1)
            elif event.keysym == 'space':
                self.play_button.invoke()
            elif event.keysym == 'Home':
                reset_animation()
            elif event.keysym == 'End':
                self.time_slider.set(self.total_time)
                self.update_display(self.total_time)
        
        self.root.bind('<Key>', key_handler)
        
        # ウィンドウが閉じられたときにアニメーションを停止
        def on_closing():
            if self.play_state["running"]:
                if self.play_state["job"] is not None:
                    self.root.after_cancel(self.play_state["job"])
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        # 初期化時に自動的にアニメーションを開始
        self.root.after(1000, self.start_animation)
    

    # setup_ui メソッドの最後に追加
    def start_animation(self):
        print("アニメーションを初期化します")
        self.play_state["running"] = True
        self.play_button.config(text="停止")
        self.animate_next_frame()

    # 次のフレームへアニメーション
    def animate_next_frame(self):
        if not self.play_state["running"]:
            return
        
        current_time = float(self.time_slider.get())
        new_time = current_time + self.detail_time_step
        
        # 最終時間を超えたかどうか
        if new_time > self.total_time:
            if self.loop_var.get():
                # ループする場合は最初に戻る
                new_time = 0
            else:
                # ループしない場合は停止
                self.play_button.invoke()  # 再生ボタンをクリックして停止
                return
        
        # スライダーと表示を更新
        self.time_slider.set(new_time)
        self.update_display(new_time)
        
        # 次のフレームのスケジュール
        delay = int(1000 / self.speed_var.get())
        self.play_state["job"] = self.root.after(delay, self.animate_next_frame)

    
    def update_display(self, time_val):
        """
        指定された時間に対応するフレームを表示
        
        Parameters:
        -----------
        time_val : float
            表示する時間（詳細データの時間単位）
        """
        # 詳細データのフレームインデックスを計算
        detail_frame = int(round(time_val / self.detail_time_step))
        
        # 範囲チェック
        if detail_frame >= self.n_frames_detail:
            detail_frame = self.n_frames_detail - 1
        
        # 概要データのフレームインデックスを計算
        overview_frame = int(round(detail_frame / self.time_ratio))
        
        # 範囲チェック
        if overview_frame >= self.n_frames_overview:
            overview_frame = self.n_frames_overview - 1
        
        # データの表示更新
        if 0 <= detail_frame < self.n_frames_detail:
            self.im_detail.set_array(self.data_detail[detail_frame])
            self.title_detail.set_text(f'(t = {time_val:.2f})')
        
        if 0 <= overview_frame < self.n_frames_overview:
            self.im_overview.set_array(self.data_overview[overview_frame])
            self.title_overview.set_text(f'(t = {time_val:.2f})')
        
        # ラベル更新
        self.time_label.config(text=f"t: {time_val:.2f} / {self.total_time:.2f}")
        self.detail_frame_label.config(text=f"frame: {detail_frame} / {self.n_frames_detail-1}")
        self.overview_frame_label.config(text=f"frame: {overview_frame} / {self.n_frames_overview-1}")
        
        # エントリーフィールド更新
        self.time_var.set(f"{time_val:.2f}")
        self.detail_frame_var.set(str(detail_frame))
        self.overview_frame_var.set(str(overview_frame))
        
        # 描画更新
        self.canvas.draw_idle()


def load_3d_data(filename):
    """3次元データ（時間, 縦, 横）のnpyファイルを読み込む"""
    try:
        data = np.load(filename)
        print(f"データの形状: {data.shape}")
        
        # データの形状を確認し、必要に応じて転置
        if len(data.shape) == 3:
            # 最後の次元が時間と仮定（縦, 横, 時間）
            if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
                data = np.transpose(data, (2, 0, 1))
                print(f"データを（時間, 縦, 横）の形式に変換しました: {data.shape}")
        
        return data
    
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

# ダミーデータの生成（ファイルが見つからない場合の対策）
def create_dummy_data():
    print("ダミーデータを生成します")
    # 3Dデータ（時間, 縦, 横）
    data_detail = np.random.random((100, 50, 80)) * 2 - 1  # -1から1の範囲の乱数
    data_overview = np.random.random((20, 25, 40)) * 2 - 1
    return data_detail, data_overview



def main():
    try: # main関数全体を try...except で囲む
        root = tk.Tk()

        detail_filename = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series.npy"
        detail_filename2 = r"C:\Users\manat\project2\surface_wave_2d\T3\T3_series.npy"
        overview_filename = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T1_series_050.npy"
        overview_filename2 = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T3_series_050.npy"

        print("--- Loading Data ---")
        data_detail1 = load_3d_data(detail_filename)
        if data_detail1 is None:
            print(f"Error: Failed to load {detail_filename}")
            return
        print(f"Shape of data_detail1: {data_detail1.shape}")

        data_detail2 = load_3d_data(detail_filename2)
        if data_detail2 is None:
            print(f"Error: Failed to load {detail_filename2}")
            return
        print(f"Shape of data_detail2: {data_detail2.shape}")

        data_overview1 = load_3d_data(overview_filename)
        if data_overview1 is None:
            print(f"Error: Failed to load {overview_filename}")
            return
        print(f"Shape of data_overview1: {data_overview1.shape}")

        data_overview2 = load_3d_data(overview_filename2)
        if data_overview2 is None:
            print(f"Error: Failed to load {overview_filename2}")
            return
        print(f"Shape of data_overview2: {data_overview2.shape}")

        print("\n--- Combining Data ---")
        try:
            print("Combining detail data...")
            # ★★★形状が一致するかチェック★★★
            if data_detail1.shape != data_detail2.shape:
                 raise ValueError(f"Shapes of detail data do not match: {data_detail1.shape} vs {data_detail2.shape}")
            data_detail = data_detail1 + data_detail2
            print(f"Shape of combined data_detail: {data_detail.shape}")
        except ValueError as e:
            print(f"Error combining detail data: {e}")
            return # エラーが発生したら終了

        try:
            print("Combining overview data...")
            # ★★★形状が一致するかチェック★★★
            if data_overview1.shape != data_overview2.shape:
                raise ValueError(f"Shapes of overview data do not match: {data_overview1.shape} vs {data_overview2.shape}")
            data_overview = data_overview1 + data_overview2
            print(f"Shape of combined data_overview: {data_overview.shape}")
        except ValueError as e:
            print(f"Error combining overview data: {e}")
            return # エラーが発生したら終了

        print("\n--- Setting up Viewer ---")

        

        detail_time_step = 7.1282897372711e-10
        overview_time_step = 3.562362796201232e-09

        viewer = DualTimeScaleViewer(
            root,
            data_detail,
            data_overview,
            detail_time_step=detail_time_step,
            overview_time_step=overview_time_step
        )
        print("Viewer initialized.")

        print("\n--- Starting mainloop ---")
        root.mainloop()
        print("Mainloop finished.") # 通常、ウィンドウを閉じるまでここには到達しない

    except Exception as e:
        print("\n--- An unexpected error occurred ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc() # 詳細なトレースバックを出力

if __name__ == "__main__":
    main()