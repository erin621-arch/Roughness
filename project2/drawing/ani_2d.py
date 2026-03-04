import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import os

def load_3d_data(filename):
    """3次元データ（縦, 横, 時間）のnpyファイルを読み込む"""
    try:
        data = np.load(filename)
        print(f"データの形状: {data.shape}")
        
        if len(data.shape) == 3:
            # 最後の次元が時間と仮定（縦, 横, 時間）
            if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
                data = np.transpose(data, (2, 0, 1))
                print(f"データを（時間, 縦, 横）の形式に変換しました: {data.shape}")
        
        return data
    
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        try:
            # 読み込みに失敗した場合はテキスト形式で試す
            data = np.loadtxt(filename)
            print(f"テキスト形式として読み込みました: {data.shape}")
            return data
        except Exception as e2:
            print(f"テキスト読み込みエラー: {e2}")
            return None

def main():
    # データ読み込み
    filename = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series.npy"  # 適宜変更
    filename2 = r"C:\Users\manat\project2\surface_wave_2d\T3\T3_series.npy"  # 適宜変更
    data = load_3d_data(filename) + load_3d_data(filename2)
    
    if data is None:
        print("データの読み込みに失敗しました。")
        return
    
    # 時間、縦、横の次元を取得
    n_frames, height, width = data.shape
    print(f"フレーム数: {n_frames}, 高さ: {height}, 幅: {width}")
    
    # 縦横比を計算（目標は1:20）
    aspect_ratio = width / height
    print(f"データの縦横比: {aspect_ratio:.2f}:1")
    
    # tkinterのルートウィンドウを作成
    root = tk.Tk()
    root.title(f"3D データビューア - 縦横比 {aspect_ratio:.2f}:1")
    
    # ウィンドウサイズを設定（1:20の縦横比に近い値）
    window_width = 2000
    window_height = 500  # ウィジェット領域を考慮
    root.geometry(f"{window_width}x{window_height}")
    
    # メインフレームを作成
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # グラフ表示用フレーム
    graph_frame = ttk.Frame(main_frame)
    graph_frame.pack(fill=tk.BOTH, expand=True)
    
    # matplotlibのfigureを作成
    # 縦横比1:20を考慮したサイズ設定
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    
    
    fig = plt.Figure(figsize=(fig_width, fig_height), dpi=100)
    ax = fig.add_subplot(111)
    
    # カラースケールの範囲
    vmin = -0.5
    vmax = 0.5
    
    # 画像表示（aspect='auto'で表示領域に合わせる）
    im = ax.imshow(data[0], cmap='jet', vmin=vmin, vmax=vmax, 
                   aspect='auto', interpolation='bilinear')
    
    # カラーバーを横向きに配置して小さめに
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.12, 
                        fraction=0.05, aspect=40)
    cbar.set_label('value')
    
    # タイトル
    title = ax.set_title(f't = 0 / {n_frames-1}')
    
    # 目盛りを表示
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Y軸の目盛りを減らして見やすくする
    ax.set_yticks(np.linspace(0, height-1, min(height, 5)).astype(int))
    
    # X軸の目盛りも適度に調整
    num_xticks = min(width, 20)  # 幅が広いので多めの目盛り
    ax.set_xticks(np.linspace(0, width-1, num_xticks).astype(int))
    
    # 必要に応じて目盛りラベルを回転
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # 余白を調整してプロット領域を最大化
    fig.tight_layout()
    
    # matplotlibのfigureをtkinterに埋め込む
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # ナビゲーションツールバーを追加（ズーム機能など）
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # コントロール用フレーム
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=5)
    
    # 現在のフレーム表示
    frame_label = ttk.Label(control_frame, text=f"フレーム: 0 / {n_frames-1}")
    frame_label.pack(side=tk.LEFT, padx=5)
    
    # スライダー変更時の処理
    def update_plot(val):
        try:
            frame = int(float(val))
            if 0 <= frame < n_frames:
                im.set_array(data[frame])
                title.set_text(f't = {frame} / {n_frames-1}')
                frame_label.config(text=f"フレーム: {frame} / {n_frames-1}")
                canvas.draw_idle()
        except:
            pass
    
    # スライダーを作成
    time_slider = ttk.Scale(
        control_frame,
        from_=0,
        to=n_frames-1,
        orient=tk.HORIZONTAL,
        command=update_plot,
        length=window_width-200
    )
    time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
    
    # 現在値を表示するエントリフィールド
    time_var = tk.StringVar(value="0")
    time_entry = ttk.Entry(control_frame, textvariable=time_var, width=6)
    time_entry.pack(side=tk.LEFT, padx=5)
    
    # エントリフィールド更新時の処理
    def entry_update(event=None):
        try:
            val = int(time_var.get())
            if 0 <= val < n_frames:
                time_slider.set(val)
                update_plot(val)
        except:
            pass
    
    time_entry.bind('<Return>', entry_update)
    
    # ボタン用フレーム
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=5)
    
    # 再生/停止用の変数と関数
    play_state = {"running": False, "job": None}
    
    # 再生ボタンのコールバック
    def play_animation():
        if play_state["running"]:
            # 停止
            if play_state["job"] is not None:
                root.after_cancel(play_state["job"])
                play_state["job"] = None
            play_state["running"] = False
            play_button.config(text="再生")
        else:
            # 再生
            play_state["running"] = True
            play_button.config(text="停止")
            animate_next_frame()
    
    # フレーム毎のアニメーション（自動再生しないよう修正）
    def animate_next_frame():
        if not play_state["running"]:
            return
        
        current = int(time_slider.get())
        next_frame = current + 1
        
        # 最終フレームに達したら停止
        if next_frame >= n_frames:
            next_frame = 0
            if not loop_var.get():  # ループしない設定なら停止
                play_button.invoke()  # 再生ボタンをクリックして停止
                return
        
        time_slider.set(next_frame)
        time_var.set(str(next_frame))
        update_plot(next_frame)
        
        # フレームレートをスライダーから取得
        delay = int(1000 / speed_var.get())
        play_state["job"] = root.after(delay, animate_next_frame)
    
    # 再生/停止ボタン
    play_button = ttk.Button(button_frame, text="再生", command=play_animation)
    play_button.pack(side=tk.LEFT, padx=5)
    
    # 前のフレームボタン
    def prev_frame():
        current = int(time_slider.get())
        if current > 0:
            time_slider.set(current - 1)
            time_var.set(str(current - 1))
    
    prev_button = ttk.Button(button_frame, text="前へ", command=prev_frame)
    prev_button.pack(side=tk.LEFT, padx=5)
    
    # 次のフレームボタン
    def next_frame():
        current = int(time_slider.get())
        if current < n_frames - 1:
            time_slider.set(current + 1)
            time_var.set(str(current + 1))
    
    next_button = ttk.Button(button_frame, text="次へ", command=next_frame)
    next_button.pack(side=tk.LEFT, padx=5)
    
    # リセットボタン
    def reset_animation():
        time_slider.set(0)
        time_var.set("0")
    
    reset_button = ttk.Button(button_frame, text="リセット", command=reset_animation)
    reset_button.pack(side=tk.LEFT, padx=5)
    
    # ループ再生チェックボックス
    loop_var = tk.BooleanVar(value=True)
    loop_check = ttk.Checkbutton(button_frame, text="ループ再生", variable=loop_var)
    loop_check.pack(side=tk.LEFT, padx=10)
    
    # 速度調整用のラベルとスライダー
    ttk.Label(button_frame, text="速度:").pack(side=tk.LEFT, padx=(10, 5))
    
    speed_var = tk.DoubleVar(value=100.0)  # デフォルト15fps
    speed_slider = ttk.Scale(
        button_frame,
        from_=100,
        to=1000,
        orient=tk.HORIZONTAL,
        variable=speed_var,
        length=150
    )
    speed_slider.pack(side=tk.LEFT)
    
    # 現在のFPS表示
    fps_label = ttk.Label(button_frame, text="15 fps")
    fps_label.pack(side=tk.LEFT, padx=5)
    
    # FPS更新
    def update_fps(event=None):
        fps = speed_var.get()
        fps_label.config(text=f"{int(fps)} fps")
    
    speed_slider.bind("<Motion>", update_fps)
    
    # キーボードショートカット
    def key_handler(event):
        if event.keysym == 'Right':
            next_frame()
        elif event.keysym == 'Left':
            prev_frame()
        elif event.keysym == 'space':
            play_button.invoke()
        elif event.keysym == 'Home':
            reset_animation()
    
    root.bind('<Key>', key_handler)
    
    # ウィンドウが閉じられたときにアニメーションを停止
    def on_closing():
        if play_state["running"]:
            if play_state["job"] is not None:
                root.after_cancel(play_state["job"])
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # tkinterのメインループを開始
    root.mainloop()

if __name__ == "__main__":
    main()