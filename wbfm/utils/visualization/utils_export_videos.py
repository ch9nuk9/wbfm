from pathlib import Path
from typing import Union

import cv2
import numpy as np
import zarr
from imutils import MicroscopeDataReader
from matplotlib import animation, pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tifffile import tifffile
from tqdm.auto import tqdm

from wbfm.utils.general.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi
from wbfm.utils.projects.finished_project_data import ProjectData


def save_video_of_neuron_trace(project_data: ProjectData, neuron_name, t0=0, t1=None, fps=7,
                               line_cmap_vec: np.array=None,
                               to_save=True, shade_kwargs=None):
    """
    Save a video of the trace for a single neuron.

    Default fps is 7, which is twice the real-time speed of the video.

    Note that t0 and t1 are in frames, not seconds.

    Parameters
    ----------
    project_data
    neuron_name
    t0
    t1
    line_cmap_vec

    Returns
    -------

    """

    # Get the data for this neuron
    if shade_kwargs is None:
        shade_kwargs = {}
    df_traces = project_data.calc_default_traces()
    y = df_traces[neuron_name]

    # Get the time vector
    if t1 is None:
        t1 = project_data.num_frames

    t = project_data.x_for_plots[t0:t1]
    y = y[t0:t1]
    if line_cmap_vec is not None:
        line_cmap_vec = line_cmap_vec[t0:t1]

    # Create the filename for the video
    vis_cfg = project_data.project_config.get_visualization_config()
    fname = f"{neuron_name}_trace_{t0}-{t1}.mp4"
    fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

    # Plot the initial trace
    fig, ax = plt.subplots(dpi=200)
    ax.set_xlabel(project_data.x_label_for_plots)
    ax.set_ylabel("Fluorescence (dR/R0)")
    ax.set_title(f"Trace for {neuron_name}")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y.min(), y.max())
    ax.plot(t, y, lw=2)

    project_data.shade_axis_using_behavior(**shade_kwargs)

    # Set up an animation to plot the trace with a vertical line moving across time
    line = ax.axvline(x=t0, color='k', lw=2)

    # def init():
    #     # line.set_data([], [])
    #     return line,

    def animate(i):
        x = t[i]
        # y = df_traces[neuron_name][x]
        line.set_xdata(x)
        if line_cmap_vec is not None:
            line.set_color(line_cmap_vec[i])
        return line,

    # Save the animation as a video
    if to_save:
        anim = animation.FuncAnimation(fig, animate, frames=t1-t0, interval=20, blit=True)
        anim.save(fname, fps=fps, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)
    print(f"Saved video of trace for neuron {neuron_name} to {fname}")

    return fname


def save_video_of_behavior(project_data: ProjectData, t0=0, t1=None, fps=7,
                           to_save=True):
    """
    Save a video of the behavior, i.e. IR video

    Parameters
    ----------
    project_data
    t0
    t1

    Returns
    -------

    """

    if t1 is None:
        t1 = project_data.num_frames

    video_fname = project_data.worm_posture_class.behavior_video_btf_fname()
    # Note that this reads the entire behavioral video into memory if it wasn't chunked
    store = tifffile.imread(video_fname, aszarr=True)
    video_zarr = zarr.open(store, mode='r')
    # Subset video to be the same fps as the fluorescence
    video_zarr = video_zarr[::project_data.physical_unit_conversion.frames_per_volume, :, :]

    if to_save:
        vis_cfg = project_data.project_config.get_visualization_config()
        fname = f"behavior_{t0}-{t1}.avi"
        fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

        write_numpy_as_avi(video_zarr[t0:t1, ...], fname, fps=fps)
        print(f"Saved video of behavior to {fname}")


def save_video_of_heatmap_with_behavior(project_path: Union[str, Path], output_fname=None):
    """
    Save a video of the heatmap (bottom half) with the behavior (on top)

    Units will not be correct unless the project_config.yaml exposure_time is set correctly

    Parameters
    ----------
    project_path

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_path, verbose=0)

    if output_fname is None:
        output_fname = project_data.project_config.get_visualization_config().resolve_relative_path("heatmap_with_behavior.mp4")

    # Get raw data
    df_traces = project_data.calc_default_traces(use_paper_options=True)

    behavior_parent_folder, behavior_raw_folder, behavior_output_folder, \
        background_img, background_video, btf_file = project_data.project_config.get_folders_for_behavior_pipeline()
    video = MicroscopeDataReader(btf_file, as_raw_tiff=True)
    video_array = video.dask_array.squeeze()

    frames_per_volume = project_data.physical_unit_conversion.frames_per_volume
    volumes_per_second = project_data.physical_unit_conversion.volumes_per_second
    um_per_pixel = project_data.physical_unit_conversion.zimmer_behavior_um_per_pixel_xy

    # Sort traces by pc1
    pc1_weights = project_data.calc_pca_modes(n_components=1, return_pca_weights=True, use_paper_options=True)
    heatmap_data = df_traces.T.reindex(pc1_weights.sort_values(by=0, ascending=False).index)

    # Get video properties
    fps = volumes_per_second
    frame_count, width, height = video_array.shape

    # Prepare VideoWriter to save the output
    output_size = (width, height * 2)  # Double the height to accommodate the heatmap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_fname, fourcc, fps, output_size)

    plot_kwargs = dict()
    plot_kwargs['vmin'] = np.quantile(df_traces.values, 0.1)
    plot_kwargs['vmax'] = np.quantile(df_traces.values, 0.99)

    # Initialize heatmap and line
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    heatmap = ax.imshow(heatmap_data, cmap='jet', interpolation='nearest', aspect='auto',
                        extent=[0, np.max(heatmap_data.T.index), 0, height], **plot_kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons")
    ax.set_yticks([])
    vertical_line = ax.axvline(x=0, color='white', linewidth=2)
    canvas = FigureCanvas(fig)
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(r'$\Delta R/R50$')

    plt.tight_layout()

    # Normalize the 2D array to the range [0, 255] for grayscale
    def normalize_to_grayscale(data):
        norm_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        return norm_data.astype(np.uint8)

    # Convert the normalized grayscale data to 3D RGB format
    def convert_to_rgb(data):
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

    # Define scale bar parameters
    scale_length_um = 100  # 1mm
    scale_length_px = int(scale_length_um / um_per_pixel)  # Convert mm to pixels
    scale_bar_start = (10, height - 50)  # Starting position (x, y)
    scale_bar_end = (10 + scale_length_px, height - 50)  # Ending position (x, y)

    # Loop through each frame in the video
    num_frames = heatmap_data.shape[1]
    # num_frames = 2
    subsample_rate = 1
    for frame_idx in tqdm(range(0, num_frames, subsample_rate)):
        # Get the video frame from dask array
        frame_idx_behavior = frame_idx * frames_per_volume
        grayscale_frame = normalize_to_grayscale(video_array[frame_idx_behavior].compute()).T
        rgb_frame = convert_to_rgb(grayscale_frame)

        # Update the vertical line to track the current time
        # Units should be same as the 'extent' of the ax.imshow command, which is the x axis of the heatmap
        line_position = heatmap_data.T.index[frame_idx]
        vertical_line.set_xdata([line_position])

        # Render the updated figure to a numpy array
        canvas.draw()
        heatmap_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_image = heatmap_image.reshape(canvas.get_width_height()[::-1] + (3,))
        # Convert heatmap from RGB to BGR to align with OpenCV's format
        heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)

        # Combine the video frame and heatmap, converting to width/height like opencv expects
        combined_frame = np.vstack((rgb_frame, heatmap_image))

        # Add text label and line for scale bar
        cv2.line(combined_frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)
        cv2.putText(combined_frame, f'{scale_length_um} um', (10 + scale_length_px // 2 - 20, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Add timestamp annotation
        timestamp = f"Time: {frame_idx / fps:.2f} s"
        cv2.putText(combined_frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the combined frame to the output video
        output_video.write(combined_frame.astype(np.uint8))

    # Release everything when done
    output_video.release()
