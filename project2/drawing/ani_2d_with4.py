import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import time


def load_and_visualize_npy(file_path1, file_path2, file_path3, file_path4):
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
    control_area.axis("off")  # Hide axes for controls

    # カラースケールの範囲
    vmin = -1
    vmax = 1

    # Initial display of first frame with correct aspect ratios
    img1 = ax1.imshow(
        data1[:, :, 0],
        extent=(0, len(data1[0, :, 0]), 0, len(data1[:, 0, 0])),
        cmap="bwr",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    img2 = ax2.imshow(
        data2[:, :, 0],
        extent=(0, len(data1[0, :, 0]), 0, len(data1[:, 0, 0])),
        cmap="bwr",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    img3 = ax3.imshow(data3[:, :, 0], cmap="bwr", aspect="auto")

    ax1.set_title(f"T1 ({height1}×{width1}×{time_steps1})")
    ax2.set_title(f"T3 ({height2}×{width2}×{time_steps2})")
    ax3.set_title(f"Overview ({height3}×{width3}×{time_steps3})")

    # Add time information
    time_text1 = ax1.text(
        0.05,
        0.95,
        f"Frame: 0/{time_steps1 - 1}",
        transform=ax1.transAxes,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    time_text2 = ax2.text(
        0.05,
        0.95,
        f"Frame: 0/{time_steps2 - 1}",
        transform=ax2.transAxes,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    time_text3 = ax3.text(
        0.05,
        0.95,
        f"Frame: 0/{time_steps3 - 1}",
        transform=ax3.transAxes,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )

    # Add colorbar for each plot
    plt.figure(fig_main.number)
    plt.colorbar(img1, ax=ax1, shrink=0.7)
    plt.colorbar(img2, ax=ax2, shrink=0.7)

    plt.figure(fig_control.number)
    plt.colorbar(img3, ax=ax3, shrink=0.7)

    # Control panel elements (in the control window)
    # Create slider
    slider_ax = plt.axes([0.15, 0.35, 0.7, 0.05])
    slider = Slider(slider_ax, "Time", 0, 1, valinit=0)
    current_time_val = 0  # Normalized time value (0-1)

    # Create animation control variables
    animation_running = False
    ani = None
    fps = 100  # Default FPS

    # Create control buttons - arranged in a grid
    play_ax = plt.axes([0.15, 0.2, 0.15, 0.08])
    play_button = Button(play_ax, "Play")

    pause_ax = plt.axes([0.35, 0.2, 0.15, 0.08])
    pause_button = Button(pause_ax, "Pause")

    next_ax = plt.axes([0.55, 0.2, 0.15, 0.08])
    next_button = Button(next_ax, "Next")

    prev_ax = plt.axes([0.75, 0.2, 0.15, 0.08])
    prev_button = Button(prev_ax, "Prev")

    # Create FPS control
    fps_ax = plt.axes([0.35, 0.05, 0.3, 0.08])
    fps_textbox = TextBox(fps_ax, "FPS: ", initial=str(fps))

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
        time_text1.set_text(f"Frame: {t1}/{time_steps1 - 1}")
        time_text2.set_text(f"Frame: {t2}/{time_steps2 - 1}")
        time_text3.set_text(f"Frame: {t3}/{time_steps3 - 1}")

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
    detail_filename = r"C:\Users\manat\project2\surface_wave_2d\T1\T1_series_middle_moremorerange_moredepth.npy"
    detail_filename2 = r"C:\Users\manat\project2\surface_wave_2d\T3\T3_series_middle_moremorerange_moredepth.npy"
    overview_filename = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T1_series_050_morerange_moredepth.npy"
    overview_filename2 = r"C:\Users\manat\project2\surface_wave_2d\050_whole\T3_series_050_morerange_moredepth.npy"

    load_and_visualize_npy(detail_filename, detail_filename2, overview_filename, overview_filename2)

    # _4はきずちゅうしんから250のやつ
