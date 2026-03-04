import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
plt.rcParams['font.size'] = 14

def visualize_fdtd_with_boundaries(file_path_t1, file_path_t3, time_frame=0, save_fig=None, times=0,
                                 # 調整可能なパラメータ
                                 f_width=0.25e-3, f_pitch=1.25e-3, f_depth=0.20e-3,
                                 probe_y_center=2033.5, 
                                 plot_y_start=1783.5, plot_y_end=2283.5,
                                 mesh_length=1.0e-5,
                                 create_colorbar_fig=False):
    """
    FDTDシミュレーション結果を境界線とプローブ中心点付きで可視化
    
    Parameters:
    -----------
    file_path_t1 : str
        T1データファイルのパス
    file_path_t3 : str  
        T3データファイルのパス
    time_frame : int
        表示する時間フレーム (default: 0)
    save_fig : str, optional
        保存するファイル名 (None の場合は表示のみ)
    create_colorbar_fig : bool
        カラーバー専用の図を作成するか (default: False)
    
    調整可能なパラメータ:
    f_width : float
        亀裂の幅 [m] (default: 0.25e-3)
    f_pitch : float
        亀裂の間隔 [m] (default: 1.25e-3)
    f_depth : float
        亀裂の深さ [m] (default: 0.20e-3)
    probe_y_center : float
        プローブ中心のy座標 [元座標系] (default: 2033.5)
    plot_y_start : float
        描画範囲の開始y座標 [元座標系] (default: 1783.5)
    plot_y_end : float
        描画範囲の終了y座標 [元座標系] (default: 2283.5)
    mesh_length : float
        メッシュサイズ [m] (default: 1.0e-5)
    """
    affr = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    # FDTDパラメータ
    yohaku = 0.0005       # 余白 m
    
    # 描画範囲の計算
    sy_center = probe_y_center  # 元の座標系での中心
    plot_width = plot_y_end - plot_y_start  # 描画範囲の幅
    probe_center_in_plot = probe_y_center - plot_y_start  # プロット内での中心位置
    
    # 離散化パラメータ
    mn_w = int(f_width / mesh_length)    # 亀裂幅の離散点数
    mn_p = int(f_pitch / mesh_length)    # 1ピッチの離散点数
    mn_nf = int((f_pitch - f_width) / mesh_length)  # 亀裂のない部分の離散点数
    mn_d = int(f_depth / mesh_length)    # 亀裂深さの離散点数
    
    print(f"亀裂パラメータ:")
    print(f"  幅: {f_width*1000:.2f} mm ({mn_w} points)")
    print(f"  間隔: {f_pitch*1000:.2f} mm ({mn_p} points)")
    print(f"  深さ: {f_depth*1000:.2f} mm ({mn_d} points)")
    
    # データ読み込み
    try:
        data_t1 = np.load(file_path_t1)
        data_t3 = np.load(file_path_t3)
        print(f"データ形状: T1={data_t1.shape}, T3={data_t3.shape}")
    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}")
        return
        
    # 時間フレームのチェック
    max_frames = min(data_t1.shape[2], data_t3.shape[2])
    if time_frame >= max_frames:
        print(f"警告: 指定された時間フレーム {time_frame} は最大値 {max_frames-1} を超えています")
        time_frame = max_frames - 1
    
    # データ取得
    t1_frame = data_t1[:, :, time_frame]
    t3_frame = data_t3[:, :, time_frame]
    
    height, width = t1_frame.shape
    y_range = width  # 500 points (sy-250:sy+250)
    
    # 亀裂境界の計算
    def get_crack_boundaries(width, height):
        """亀裂の境界線座標を計算"""
        boundaries = []
        
        # y方向の亀裂数を計算（描画範囲内）
        # 元の座標系での描画範囲: plot_y_start to plot_y_end
        start_y_original = plot_y_start
        end_y_original = plot_y_end
        
        # 亀裂の開始位置を計算
        num_cracks_in_range = 0
        crack_positions = []
        
        # 最初の亀裂位置を見つける
        first_crack_start = mn_nf  # 最初の亀裂の開始位置
        
        i = 0
        while True:
            crack_start_global = mn_nf + i * mn_p
            crack_end_global = crack_start_global + mn_w
            
            # 描画範囲との重複をチェック
            crack_start_in_plot = crack_start_global - start_y_original
            crack_end_in_plot = crack_end_global - start_y_original
            
            # 範囲外なら終了
            if crack_start_global > end_y_original:
                break
                
            # 範囲内に一部でも含まれていれば追加
            if crack_end_global > start_y_original:
                # プロット座標系に変換
                plot_start = max(0, crack_start_in_plot)
                plot_end = min(width, crack_end_in_plot)
                
                if plot_start < width and plot_end > 0:
                    crack_positions.append((plot_start, plot_end))
                    num_cracks_in_range += 1
            
            i += 1
            
        print(f"描画範囲内の亀裂数: {num_cracks_in_range}")
        print(f"描画範囲: y={plot_y_start:.1f} to {plot_y_end:.1f} (元座標系)")
        print(f"プローブ中心: y={probe_y_center:.1f} → プロット内位置 {probe_center_in_plot:.1f}")
        
        #  境界線の座標を生成
        for crack_start, crack_end in crack_positions:
            # origin='upper'の場合、y座標を反転させる必要がある
            # 亀裂の上端（材料表面からの深さ）
            crack_top_original = height - mn_d  # 元の座標
            crack_bottom_original = height      # 元の座標
            
            # origin='upper'用に座標変換
            crack_top = mn_d      # 上端（浅い部分）
            crack_bottom = 0      # 下端（表面）
            
            # 垂直境界線（亀裂の両端）
            if crack_start >= 0:
                boundaries.append([(crack_start, crack_bottom), (crack_start, crack_top)])
            if crack_end < width:
                boundaries.append([(crack_end, crack_bottom), (crack_end, crack_top)])
            
            # 水平境界線（亀裂の底）
            boundaries.append([(crack_start, crack_top), (crack_end, crack_top)])
        
        return boundaries, crack_positions
    
    # 境界線の計算
    boundaries, crack_positions = get_crack_boundaries(width, height)
    
    # カラースケール設定
    vmin, vmax = -1.0, 1.0
    
    # 図の作成 - GridSpecで2列（プロット1, プロット2のみ）
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    # サブプロットを作成
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    # T1の描画
    im1 = ax1.imshow(t1_frame, extent=[0, width, 0, height], 
                    cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax, origin='upper')
    ax1.set_title(f'T1 - Timestep {time_frame}', fontsize=16)
    ax1.set_ylabel(r'x-axis [$\times 10^{-5}$ mm]', fontsize=14)
    ax1.set_xlabel(r'z-axis [$\times 10^{-5}$ mm]', fontsize=14)
    # 左上に (a) を追加
    plt.text(0.02, 0.98, f'({affr[times]})', transform=ax1.transAxes,
        fontsize=32, weight='bold', ha='left', va='top')

    # T3の描画  
    im2 = ax2.imshow(t3_frame, extent=[0, width, 0, height], 
                    cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax, origin='upper')
    ax2.set_title(f'T3 - Timestep {time_frame}', fontsize=16)
    ax2.set_xlabel(r'z-axis [$\times 10^{-5}$ mm]', fontsize=14)

    # ax2のy軸ラベルを非表示（sharey=Trueなので）
    ax2.tick_params(labelleft=False)

    # 境界線の描画
    for boundary in boundaries:
        (x1, y1), (x2, y2) = boundary
        ax1.plot([x1, x2], [y1, y2], 'black', linewidth=2, alpha=0.8)
        ax2.plot([x1, x2], [y1, y2], 'black', linewidth=2, alpha=0.8)

    # プローブ中心線の描画
    probe_x = probe_center_in_plot

    # 各プロット内にも中心線を表示
    for ax in [ax1, ax2]:
        ax.axvline(x=probe_x, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Center of probe')
        ax.legend(loc='upper right')
    
    # 保存または表示
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save_fig}")
    else:
        plt.show()
    
    # カラーバー専用の図を作成
    if create_colorbar_fig:
        # 元の図の高さを取得
        original_height = 6
        colorbar_height = original_height * 3  # 縦に3倍
        
        # カラーバー専用の図を作成
        fig_colorbar = plt.figure(figsize=(2, colorbar_height))
        
        # ダミーのimshowを作成してカラーバーを生成
        dummy_data = np.array([[vmin, vmax]])
        im_dummy = plt.imshow(dummy_data, cmap='bwr', vmin=vmin, vmax=vmax)
        plt.gca().set_visible(False)  # 軸を非表示
        
        # カラーバーを作成
        cbar = plt.colorbar(im_dummy, fraction=1.0, aspect=20)
        cbar.set_label('[Pa]', rotation=0, labelpad=20, fontsize=14)
        
        # カラーバー専用ファイルとして保存
        if save_fig:
            colorbar_filename = save_fig.replace('.jpg', '_colorbar.jpg').replace('.eps', '_colorbar.eps').replace('.png', '_colorbar.png')
            plt.savefig(colorbar_filename, dpi=300, bbox_inches='tight')
            print(f"カラーバー図を保存しました: {colorbar_filename}")
        else:
            plt.show()


def create_standalone_colorbar(vmin=-1.0, vmax=1.0, height_multiplier=3, save_fig=None):
    """
    カラーバーのみを含む図を作成する関数
    
    Parameters:
    -----------
    vmin : float
        カラースケールの最小値
    vmax : float
        カラースケールの最大値
    height_multiplier : float
        高さの倍率
    save_fig : str, optional
        保存するファイル名
    """
    base_height = 6
    colorbar_height = base_height * height_multiplier
    
    fig = plt.figure(figsize=(2, colorbar_height))
    
    # ダミーのimshowを作成
    dummy_data = np.array([[vmin, vmax]])
    im_dummy = plt.imshow(dummy_data, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.gca().set_visible(False)  # 軸を非表示
    
    # カラーバーを作成
    cbar = plt.colorbar(im_dummy, fraction=1.0, aspect=20)
    cbar.set_label('[Pa]', rotation=0, labelpad=20, fontsize=14)
    
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"カラーバー図を保存しました: {save_fig}")
    else:
        plt.show()


# 使用例とパラメータ設定の例
if __name__ == "__main__":
    # ファイルパス（適宜変更してください）
    t1_file = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series_middle_moremorerange_moredepth.npy"
    t3_file = r"C:\Users\manat\project2\surface_wave_2d\T3\T3_series_middle_moremorerange_moredepth.npy"

    # 基本パラメータ（デフォルト値）
    params_default = {
        'f_width': 0.25e-3,           # 亀裂幅 0.25mm
        'f_pitch': 2.00e-3,           # 亀裂間隔 2.00mm  
        'f_depth': 0.20e-3,           # 亀裂深さ 0.20mm
        'probe_y_center': int((1799+1975)/2),     # プローブ中心 y座標
        'plot_y_start': 1750,         # 描画開始位置 
        'plot_y_end': 2250,           # 描画終了位置 
        'mesh_length': 1.0e-5         # メッシュサイズ
    }

    time = [1070, 1202, 1322, 1468, 1614, 1694, 2194, 2244]
    affr = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    # メイン図の作成（カラーバーなし）
    for i in range(len(time)):
        visualize_fdtd_with_boundaries(t1_file, t3_file, time_frame=time[i], 
                                 save_fig=fr"C:\Users\manat\project2\drawing\jasa\Figure13{affr[i]}.jpg", 
                                 times=i, **params_default)
    
    # カラーバー専用図の作成（1回だけ）
    create_standalone_colorbar(vmin=-1.0, vmax=1.0, height_multiplier=3, 
                              save_fig=r"C:\Users\manat\project2\drawing\jasa\Figure13_colorbar.jpg")