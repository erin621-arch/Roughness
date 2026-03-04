import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import time

def load_and_visualize_npy(file_path1, file_path2, file_path3, file_path4, 
                          probe_y_center=None, plot_y_start_detail=None):
    """
    Load and visualize .npy files with synchronized time control and playback features.
    Main window shows detail views, control window shows overview and controls.
    
    Parameters:
    -----------
    file_path1 : str
        Path to first detail .npy file with shape [height, width, time]
    file_path2 : str
        Path to second detail .npy file with shape [height, width, time]
    file_path3 : str
        Path to first overview .npy file with shape [height, width, time]
    file_path4 : str
        Path to second overview .npy file with shape [height, width, time]
    probe_y_center : float, optional
        Center position of probe in original coordinate system
    plot_y_start_detail : float, optional
        Start position of detail plot range in original coordinate system
    """
    # Load the data
    data1 = np.load(file_path1)  # detail_filename
    data2 = np.load(file_path2)  # detail_filename2
    data3 = np.load(file_path3) + np.load(file_path4)  # overview (combined)
    
    # Get dimensions
    height1, width1, time_steps1 = data1.shape
    height2, width2, time_steps2 = data2.shape
    height3, width3, time_steps3 = data3.shape
    
    # Create main figure for detail views
    fig_main = plt.figure(figsize=(12, 8))
    
    # Create control figure for overview and controls
    fig_control = plt.figure(figsize=(8, 8))
    
    # Main window: Detail views (2x1 layout)
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = fig_main.add_subplot(gs_main[0, 0])  # T1 detail
    ax2 = fig_main.add_subplot(gs_main[1, 0])  # T3 detail
    
    # Control window: Overview and control panel
    gs_control = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax3 = fig_control.add_subplot(gs_control[0, 0])  # Overview
    control_area = fig_control.add_subplot(gs_control[1, 0])  # Control panel
    control_area.axis('off')  # Hide axes for controls
    
    # カラースケールの範囲
    vmin = -1
    vmax = 1

    # Initial display of first frame with correct aspect ratios
    img1 = ax1.imshow(data1[:, :, 0], extent=(0, width1, 0, height1), cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax, origin='upper')
    img2 = ax2.imshow(data2[:, :, 0], extent=(0, width2, 0, height2), cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax, origin='upper')
    img3 = ax3.imshow(data3[:, :, 0], extent=(0, width3, 0, height3), cmap='bwr', aspect='auto', origin='upper')
    
    ax1.set_title(f'T1 ({height1}×{width1}×{time_steps1})')
    ax2.set_title(f'T3 ({height2}×{width2}×{time_steps2})')
    ax3.set_title(f'Overview ({height3}×{width3}×{time_steps3})')
    
    # 軸ラベル設定
    ax1.set_ylabel('x-axis [mm]')
    ax1.set_xlabel('z-axis [mm]')
    ax2.set_ylabel('x-axis [mm]')
    ax2.set_xlabel('z-axis [mm]')
    ax3.set_ylabel('x-axis [mm]')
    ax3.set_xlabel('z-axis [mm]')
    
    # 軸の目盛りラベルをmm単位に変換（×1e-2）
    def update_axis_labels(ax):
        current_xticks = ax.get_xticks()
        current_yticks = ax.get_yticks()
        ax.set_xticklabels([f'{tick*1e-2:.2f}' for tick in current_xticks])
        ax.set_yticklabels([f'{tick*1e-2:.2f}' for tick in current_yticks])
    
    update_axis_labels(ax1)
    update_axis_labels(ax2)
    update_axis_labels(ax3)
    
    # プローブ中心線を追加（凡例なし）
    # T1, T3用：描画範囲内での相対位置を計算
    if probe_y_center is not None and plot_y_start_detail is not None:
        probe_center_detail = probe_y_center - plot_y_start_detail
    else:
        probe_center_detail = width1 / 2  # デフォルトは中央
    
    # Overview用：中央
    probe_center_overview = width3 / 2
    
    ax1.axvline(x=probe_center_detail, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(x=probe_center_detail, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax3.axvline(x=probe_center_overview, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add time information
    time_text1 = ax1.text(0.05, 0.95, f'Frame: 0/{time_steps1-1}', 
                         transform=ax1.transAxes, color='white', 
                         bbox=dict(facecolor='black', alpha=0.5))
    time_text2 = ax2.text(0.05, 0.95, f'Frame: 0/{time_steps2-1}', 
                         transform=ax2.transAxes, color='white',
                         bbox=dict(facecolor='black', alpha=0.5))
    time_text3 = ax3.text(0.05, 0.95, f'Frame: 0/{time_steps3-1}', 
                         transform=ax3.transAxes, color='white',
                         bbox=dict(facecolor='black', alpha=0.5))
    
    # Add colorbar for each plot
    plt.figure(fig_main.number)
    plt.colorbar(img1, ax=ax1, shrink=0.7)
    plt.colorbar(img2, ax=ax2, shrink=0.7)
    
    plt.figure(fig_control.number)
    plt.colorbar(img3, ax=ax3, shrink=0.7)
    
    # Control panel elements (in the control window)
    # Create slider
    slider_ax = plt.axes([0.15, 0.35, 0.7, 0.05])
    slider = Slider(slider_ax, 'Time', 0, 1, valinit=0)
    current_time_val = 0  # Normalized time value (0-1)
    
    # Create animation control variables
    animation_running = False
    ani = None
    fps = 100  # Default FPS
    
    # Create control buttons - arranged in a grid
    play_ax = plt.axes([0.15, 0.2, 0.15, 0.08])
    play_button = Button(play_ax, 'Play')
    
    pause_ax = plt.axes([0.35, 0.2, 0.15, 0.08])
    pause_button = Button(pause_ax, 'Pause')
    
    next_ax = plt.axes([0.55, 0.2, 0.15, 0.08])
    next_button = Button(next_ax, 'Next')
    
    prev_ax = plt.axes([0.75, 0.2, 0.15, 0.08])
    prev_button = Button(prev_ax, 'Prev')
    
    # Create FPS control
    fps_ax = plt.axes([0.35, 0.05, 0.3, 0.08])
    fps_textbox = TextBox(fps_ax, 'FPS: ', initial=str(fps))
    
    # Update function for slider
    def update(val):
        nonlocal current_time_val
        current_time_val = val
        
        # Convert normalized slider value (0-1) to appropriate time indices for each dataset
        t1 = int(val * (time_steps1 - 1))
        t2 = int(val * (time_steps2 - 1))
        t3 = int(val * (time_steps3 - 1))
        
        # Update images
        img1.set_array(data1[:, :, t1])
        img2.set_array(data2[:, :, t2])
        img3.set_array(data3[:, :, t3])
        
        # Update time text
        time_text1.set_text(f'Frame: {t1}/{time_steps1-1}')
        time_text2.set_text(f'Frame: {t2}/{time_steps2-1}')
        time_text3.set_text(f'Frame: {t3}/{time_steps3-1}')
        
        # 軸ラベルを再更新
        update_axis_labels(ax1)
        update_axis_labels(ax2)
        update_axis_labels(ax3)
        
        # Update both figures
        fig_main.canvas.draw_idle()
        fig_control.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update)
    
    # Animation update function
    def animate(i):
        nonlocal current_time_val
        
        # Get current FPS value for step size
        try:
            current_fps = int(fps_textbox.text)
            if current_fps <= 0:
                current_fps = 1
        except ValueError:
            current_fps = fps  # Use stored value if invalid
        
        # Calculate time increment based on FPS
        # Higher FPS means larger steps per frame
        increment = (current_fps / 10.0) / max(time_steps1, time_steps2, time_steps3)
        current_time_val = min(current_time_val + increment, 1.0)
        
        # Loop back to beginning if at the end
        if current_time_val >= 1.0:
            current_time_val = 0.0
        
        # Update slider position
        slider.set_val(current_time_val)
        
        # Return the artists that were updated
        return [img1, img2, img3, time_text1, time_text2, time_text3]
    
    # Button callback functions
    def play(event):
        nonlocal animation_running, ani, fps
        
        # Cancel any existing animation
        if animation_running and ani is not None:
            ani.event_source.stop()
        
        # Get current FPS value
        try:
            fps = int(fps_textbox.text)
            if fps <= 0:
                fps = 1
        except ValueError:
            fps = 10
        
        # Calculate interval in milliseconds
        interval = 1000 / fps
        
        # Start new animation - use control figure as the base
        animation_running = True
        ani = FuncAnimation(fig_control, animate, interval=interval, blit=False)
        plt.draw()
    
    def pause(event):
        nonlocal animation_running, ani
        if animation_running and ani is not None:
            ani.event_source.stop()
            animation_running = False
    
    def next_step(event):
        nonlocal current_time_val
        
        # Get current FPS value for step size
        try:
            current_fps = int(fps_textbox.text)
            if current_fps <= 0:
                current_fps = 1
        except ValueError:
            current_fps = fps  # Use stored value if invalid
        
        # Higher FPS means larger steps
        # Calculate time increment based on FPS
        increment = (current_fps / 10.0) / max(time_steps1, time_steps2, time_steps3)
        current_time_val = min(current_time_val + increment, 1.0)
        
        # Update slider position
        slider.set_val(current_time_val)

    def prev_step(event):
        nonlocal current_time_val
        
        # Get current FPS value for step size
        try:
            current_fps = int(fps_textbox.text)
            if current_fps <= 0:
                current_fps = 1
        except ValueError:
            current_fps = fps  # Use stored value if invalid
        
        # Calculate time decrement based on FPS
        decrement = (current_fps / 10.0) / max(time_steps1, time_steps2, time_steps3)
        current_time_val = max(current_time_val - decrement, 0.0)
        
        # Update slider position
        slider.set_val(current_time_val)
    
    def update_fps(text):
        nonlocal fps, animation_running, ani
        
        # Update FPS value
        try:
            new_fps = int(text)
            if new_fps <= 0:
                new_fps = 1
            fps = new_fps
        except ValueError:
            return
        
        # Restart animation with new FPS if running
        if animation_running:
            if ani is not None:
                ani.event_source.stop()
            play(None)
    
    # Connect button callbacks
    play_button.on_clicked(play)
    pause_button.on_clicked(pause)
    next_button.on_clicked(next_step)
    prev_button.on_clicked(prev_step)
    fps_textbox.on_submit(update_fps)
    
    # Adjust layouts
    plt.figure(fig_main.number)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    plt.figure(fig_control.number)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=0.95)
    
    # Show both windows
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    t1_file = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series_middle_moremorerange_moredepth.npy"
    t3_file = r"C:\Users\manat\project2\surface_wave_2d\T3\T3_series_middle_moremorerange_moredepth.npy"
    overview_filename = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T1_series_050_morerange_moredepth.npy"
    overview_filename2 = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T3_series_050_morerange_moredepth.npy"

    # プローブ中心位置の設定（seisiga copyと同じパラメータ）
    probe_y_center = int((1799+1975)/2)  # プローブ中心のy座標
    plot_y_start_detail = 1750  # 詳細ビューの描画開始位置

    load_and_visualize_npy(t1_file, t3_file, overview_filename, overview_filename2,
                          probe_y_center=probe_y_center, plot_y_start_detail=plot_y_start_detail)

    #_4はきとめちゅうしんから250のやつ