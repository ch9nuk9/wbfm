import numpy as np
import zarr
from matplotlib import animation, pyplot as plt
from tifffile import tifffile

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

    video_fname = project_data.worm_posture_class.behavior_video_avi_fname()
    video_fname = video_fname.with_suffix('.btf')
    # Note that this reads the entire behavioral video into memory if it wasn't chunked
    store = tifffile.imread(video_fname, aszarr=True)
    video_zarr = zarr.open(store, mode='r')
    # Subset video to be the same fps as the fluorescence
    video_zarr = video_zarr[::project_data.worm_posture_class.frames_per_volume, :, :]

    if to_save:
        vis_cfg = project_data.project_config.get_visualization_config()
        fname = f"behavior_{t0}-{t1}.avi"
        fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

        write_numpy_as_avi(video_zarr[t0:t1, ...], fname, fps=fps)
        print(f"Saved video of behavior to {fname}")
