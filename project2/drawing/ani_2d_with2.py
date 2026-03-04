import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import time

def load_and_visualize_npy(file_path1, file_path2, file_path3, file_path4):
    """
    Load and visualize two .npy files with synchronized time control and playback features.
    
    Parameters:
    -----------
    file_path1 : str
        Path to first .npy file with shape [height, width, time]
    file_path2 : str
        Path to second .npy file with shape [height, width, time]
    """
    # Load the data
    data1 = np.load(file_path1) + np.load(file_path2)
    data2 = np.load(file_path3) + np.load(file_path4)
    
    # Get dimensions
    height1, width1, time_steps1 = data1.shape
    height2, width2, time_steps2 = data2.shape
    
    # Create figure with vertical layout
    fig = plt.figure(figsize=(12, 10))
    
    # Create GridSpec for vertical layout with controls at bottom
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 0.2, 0.3])
    
    # Create axes for each dataset
    ax1 = fig.add_subplot(gs[0])  # 上部3/4のスペース
    ax2 = fig.add_subplot(gs[1])  # 下部1/4のスペース
    
    # Create slider axis
    ax_slider = fig.add_subplot(gs[2])
    
    # Create control buttons area
    ax_controls = fig.add_subplot(gs[3])
    ax_controls.axis('off')  # Hide axes for controls
    
    # Initial display of first frame with correct aspect ratios

    # カラースケールの範囲
    vmin = -1
    vmax = 1

    img1 = ax1.imshow(data1[:, :, 0], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    img2 = ax2.imshow(data2[:, :, 0], cmap='jet', aspect='auto')
    
    ax1.set_title(f'Dataset 1 ({height1}×{width1}×{time_steps1})')
    ax2.set_title(f'Dataset 2 ({height2}×{width2}×{time_steps2})')
    
    # Add time information
    time_text1 = ax1.text(0.05, 0.95, f'Frame: 0/{time_steps1-1}', 
                         transform=ax1.transAxes, color='white', 
                         bbox=dict(facecolor='black', alpha=0.5))
    time_text2 = ax2.text(0.05, 0.95, f'Frame: 0/{time_steps2-1}', 
                         transform=ax2.transAxes, color='white',
                         bbox=dict(facecolor='black', alpha=0.5))
    
    # Add colorbar for each plot
    plt.colorbar(img1, ax=ax1, shrink=0.5)
    plt.colorbar(img2, ax=ax2, shrink=0.5)
    
    # Create slider
    slider = Slider(ax_slider, 'Time', 0, 1, valinit=0)
    current_time_val = 0  # Normalized time value (0-1)
    
    # Create animation control variables
    animation_running = False
    ani = None
    fps = 1000  # Default FPS

    # FPS indicator text
    fps_indicator = plt.figtext(0.8, 0.07, f'Current FPS: {fps}', fontsize=9)
    
    # Create control buttons
    play_ax = plt.axes([0.1, 0.15, 0.1, 0.04])
    play_button = Button(play_ax, 'Play')
    
    pause_ax = plt.axes([0.21, 0.15, 0.1, 0.04])
    pause_button = Button(pause_ax, 'Pause')
    
    next_ax = plt.axes([0.32, 0.15, 0.1, 0.04])
    next_button = Button(next_ax, 'Next')
    
    prev_ax = plt.axes([0.43, 0.15, 0.1, 0.04])
    prev_button = Button(prev_ax, 'Prev')
    
    # Create FPS control
    fps_ax = plt.axes([0.8, 0.15, 0.06, 0.04])
    fps_textbox = TextBox(fps_ax, 'FPS: ', initial=str(fps))
    
    fps_label = plt.figtext(0.54, 0.15, 'Playback Speed:', fontsize=9)
    
    # Update function for slider
    def update(val):
        nonlocal current_time_val
        current_time_val = val
        
        # Convert normalized slider value (0-1) to appropriate time indices for each dataset
        t1 = int(val * (time_steps1 - 1))
        t2 = int(val * (time_steps2 - 1))
        
        # Update images
        img1.set_array(data1[:, :, t1])
        img2.set_array(data2[:, :, t2])
        
        # Update time text
        time_text1.set_text(f'Frame: {t1}/{time_steps1-1}')
        time_text2.set_text(f'Frame: {t2}/{time_steps2-1}')
        
        # Update display
        fig.canvas.draw_idle()
    
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
        increment = (current_fps / 10.0) / max(time_steps1, time_steps2)
        current_time_val = min(current_time_val + increment, 1.0)
        
        # Loop back to beginning if at the end
        if current_time_val >= 1.0:
            current_time_val = 0.0
        
        # Update slider position
        slider.set_val(current_time_val)
        
        # Return the artists that were updated
        return [img1, img2, time_text1, time_text2]
    
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
        
        # Update FPS indicator
        fps_indicator.set_text(f'Current FPS: {fps}')
        
        # Calculate interval in milliseconds
        interval = 1000 / fps
        
        # Start new animation
        animation_running = True
        ani = FuncAnimation(fig, animate, interval=interval, blit=True)
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
        increment = (current_fps / 10.0) / max(time_steps1, time_steps2)
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
        
        # Higher FPS means larger steps
        # Calculate time decrement based on FPS
        decrement = (current_fps / 10.0) / max(time_steps1, time_steps2)
        current_time_val = max(current_time_val - decrement, 0.0)
        
        # Update slider position
        slider.set_val(current_time_val)
        # Calculate time decrement - one frame at the current FPS rate
        decrement = 1.0 / max(time_steps1, time_steps2) * (30 / current_fps)
        current_time_val = max(current_time_val - decrement, 0.0)
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
        
        # Update FPS indicator
        fps_indicator.set_text(f'Current FPS: {fps}')
        
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
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for controls
    plt.show()



# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    detail_filename = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series_middle_morerange.npy"
    detail_filename2 = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series_middle_morerange.npy"
    overview_filename = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T1_series_050_morerange.npy"
    overview_filename2 = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T3_series_050_morerange.npy"
    
    load_and_visualize_npy(detail_filename, detail_filename2, overview_filename, overview_filename2)