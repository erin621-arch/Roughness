import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os  # ディレクトリ操作用に追加

# ================== 設定パラメータ ==================
# メッシュサイズ (0.01mm)
mesh_length = 1.0e-5 

# 表示したいステップサイズのリスト
step_sizes = [1, 2, 5, 10]

# 各形状のテスト用パラメータ
# くさび
kusabi_pitch = 1.25e-3  
kusabi_depth = 0.20e-3 

# 三角
sankaku_width = 0.25e-3 #(固定)
sankaku_depth = 0.20e-3 

# 半円(U字)
ushape_width = 0.25e-3 #(固定)
ushape_depth = 0.20e-3

# ★ 保存先ディレクトリ設定
# SAVE_DIR = r"C:\Users\cs16\Roughness\project4"  # 研究室PC
SAVE_DIR = r"C:/Users/hisay/OneDrive/ドキュメント/test_folder"   # 自宅PC

# ==================================================

def ensure_directory_exists(path):
    """ディレクトリが存在しない場合に作成する関数"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"ディレクトリを作成しました: {path}")
        except OSError as e:
            print(f"ディレクトリ作成エラー: {e}")

def plot_grid(ax, binary_map, title, mesh_len, max_depth_mesh, max_width_mesh):
    """グリッドとブロックを描画するヘルパー関数"""
    rows, cols = binary_map.shape
    
    # --- グリッド線の設定 ---
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    
    # グリッドを描画
    ax.grid(which='major', color='#cccccc', linestyle='-', linewidth=0.5, alpha=0.8)
    
    # 目盛りの数字は消す
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # ブロックの描画 (1の部分を塗りつぶす)
    for r in range(rows):
        for c in range(cols):
            if binary_map[r, c] == 1:
                # r:深さ(0が底面), c:横方向
                rect = patches.Rectangle((c, r), 1, 1, linewidth=0, edgecolor=None, facecolor='navy')
                ax.add_patch(rect)
    
    # 軸の設定
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    
    # mm単位の補助目盛り
    display_interval = 5
    yticks_major = np.arange(0, rows + 1, display_interval)
    yticklabels = [f"{y*mesh_len*1000:.2f}" for y in yticks_major]
    
    ax2 = ax.secondary_yaxis('left')
    ax2.set_yticks(yticks_major)
    ax2.set_yticklabels(yticklabels, fontsize=8)
    ax2.tick_params(length=0)
    if title.startswith("Step Size = 1"):
        ax2.set_ylabel("Depth (mm)", fontsize=9)

def visualize_kusabi():
    print("Generating Kusabi visualization...")
    # ★変更点: (len(step_sizes), 1) で縦並び指定、figsizeを縦長に変更
    fig, axes = plt.subplots(len(step_sizes), 1, figsize=(8, 4 * len(step_sizes)))
    fig.suptitle(f"Wedge (Pitch={kusabi_pitch*1000}mm, Depth={kusabi_depth*1000}mm)", fontsize=14)
    
    mn_p = int(round(kusabi_pitch / mesh_length))
    mn_d = int(round(kusabi_depth / mesh_length))
    
    for idx, step_size in enumerate(step_sizes):
        ax = axes[idx]
        grid = np.zeros((mn_d + 2, mn_p + 2), dtype=int)
        
        for y in range(mn_p):
            local_y = y
            # 左端(y)での深さを計算
            ideal_depth_val = mn_d * (1.0 - local_y / mn_p)
            current_depth = (int(ideal_depth_val) // step_size) * step_size
            for d in range(current_depth):
                grid[d, y] = 1
        
        plot_grid(ax, grid, f"Step Size = {step_size}", mesh_length, mn_d, mn_p)
        
        # 理想線の描画
        ax.plot([0, mn_p], [mn_d, 0], 'r--', linewidth=1.5, label='Ideal')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # タイトルとグラフが重ならないように調整
    
    # ★変更点: 指定ディレクトリへ保存
    save_path = os.path.join(SAVE_DIR, "kusabi_vertical.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close() # メモリ解放のため閉じる（表示したい場合は plt.show() に戻してください）

def visualize_sankaku():
    print("Generating Sankaku visualization...")
    # ★変更点: 縦並び指定、figsize変更
    fig, axes = plt.subplots(len(step_sizes), 1, figsize=(8, 4 * len(step_sizes)))
    fig.suptitle(f"Sankaku (Width={sankaku_width*1000}mm, Depth={sankaku_depth*1000}mm)", fontsize=14)
    
    mn_w = int(round(sankaku_width / mesh_length))
    if mn_w % 2 == 0: mn_w -= 1
    mn_d = int(round(sankaku_depth / mesh_length))
    
    canvas_w = mn_w + 6
    canvas_h = mn_d + 2
    center_y = canvas_w // 2 
    
    for idx, step_size in enumerate(step_sizes):
        ax = axes[idx]
        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        
        for d in range(mn_d):
            d_step = (d // step_size) * step_size
            raw_w = mn_w * (1.0 - d_step / mn_d)
            width_at_d = int(round(raw_w))
            if width_at_d % 2 == 0: width_at_d -= 1
            if width_at_d < 1: width_at_d = 1
            half = width_at_d // 2
            yl = center_y - half
            yr = center_y + half + 1
            grid[d, yl:yr] = 1
        
        plot_grid(ax, grid, f"Step Size = {step_size}", mesh_length, mn_d, mn_w)
        
        visual_center = center_y + 0.5
        ax.plot([visual_center - mn_w/2, visual_center], [0, mn_d], 'r--', linewidth=1.5)
        ax.plot([visual_center + mn_w/2, visual_center], [0, mn_d], 'r--', linewidth=1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # ★変更点: 指定ディレクトリへ保存
    save_path = os.path.join(SAVE_DIR, "sankaku_vertical.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def visualize_ushape():
    print("Generating Ushape visualization...")
    fig, axes = plt.subplots(len(step_sizes), 1, figsize=(8, 4 * len(step_sizes)))
    fig.suptitle(f"U-shape (Width={ushape_width*1000}mm, Depth={ushape_depth*1000}mm)", fontsize=14)
    
    mn_w = int(round(ushape_width / mesh_length))
    if mn_w % 2 == 0: mn_w -= 1
    mn_d = int(round(ushape_depth / mesh_length))
    
    # --- グリッド生成用の整数計算 ---
    mn_r = mn_w // 2
    mn_straight = mn_d - mn_r
    
    # --- 赤線（理想線）用の正確な計算 ---
    # ★修正点: 赤線用の変数を別途定義します
    ideal_radius = mn_w / 2.0          # 正確な半径 (例: 12.5)
    ideal_straight = mn_d - ideal_radius # 正確な直線部の長さ (例: 20 - 12.5 = 7.5)

    canvas_w = mn_w + 6
    canvas_h = mn_d + 2
    center_y = canvas_w // 2
    
    for idx, step_size in enumerate(step_sizes):
        ax = axes[idx]
        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        
        # グリッドの計算（ここは元のまま変えません）
        for d in range(mn_d):
            d_step = (d // step_size) * step_size
            width_at_d = 0
            if d_step < mn_straight:
                width_at_d = mn_w
            else:
                dy = d_step - mn_straight
                if dy < mn_r:
                    val = np.sqrt(mn_r**2 - dy**2)
                    width_at_d = int(val * 2)
                else:
                    width_at_d = 0
            if width_at_d > 0 and width_at_d % 2 == 0: width_at_d -= 1
            if width_at_d < 0: width_at_d = 0
            
            half = width_at_d // 2
            yl = center_y - half
            yr = center_y + half + 1
            grid[d, yl:yr] = 1
            
        plot_grid(ax, grid, f"Step Size = {step_size}", mesh_length, mn_d, mn_w)
        
        visual_center = center_y + 0.5
        
        # --- 赤線の描画（修正済み） ---
        # ★修正点: mn_straight ではなく ideal_straight を使います
        ax.plot([visual_center - mn_w/2, visual_center - mn_w/2], [0, ideal_straight], 'r--', linewidth=1.5)
        ax.plot([visual_center + mn_w/2, visual_center + mn_w/2], [0, ideal_straight], 'r--', linewidth=1.5)
        
        theta = np.linspace(0, np.pi, 100)
        x_arc = visual_center + (mn_w/2) * np.cos(theta)
        # ★修正点: ここも ideal_straight と ideal_radius を使います
        y_arc = ideal_straight + ideal_radius * np.sin(theta)
        ax.plot(x_arc, y_arc, 'r--', linewidth=1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    save_path = os.path.join(SAVE_DIR, "ushape_vertical.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 実行時にディレクトリをチェック/作成
    ensure_directory_exists(SAVE_DIR)
    
    visualize_kusabi()
    visualize_sankaku()
    visualize_ushape()
    
    print("すべての処理が完了しました。")