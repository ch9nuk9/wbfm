import numpy as np
from matplotlib import animation, pyplot as plt


def save_video_of_single_neuron(project_data, neuron_name, t0=0, t1=None):
    """
    Save a video of the trace for a single neuron.

    Default fps is 7, which is twice the real-time speed of the video.

    Parameters
    ----------
    project_data
    neuron_name
    t0
    t1

    Returns
    -------

    """

    # Get the data for this neuron
    df_traces = project_data.calc_default_traces()
    y = df_traces[neuron_name]

    # Get the time vector
    if t1 is None:
        t1 = project_data.num_frames

    t = np.arange(t0, t1)
    y = y[t0:t1]

    # Create the filename for the video
    vis_cfg = project_data.project_config.get_visualization_config()
    fname = f"{neuron_name}_trace_{t0}-{t1}.mp4"
    fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)

    # Plot the initial trace
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Fluorescence (dR/R0)")
    ax.set_title(f"Trace for {neuron_name}")

    ax.set_xlim(t0, t1)
    ax.set_ylim(y.min(), y.max())
    ax.plot(t, y, lw=2)

    project_data.shade_axis_using_behavior()

    # Set up an animation to plot the trace with a vertical line moving across time
    line = ax.axvline(x=t0, color='k', lw=2)

    # def init():
    #     # line.set_data([], [])
    #     return line,

    def animate(i):
        x = t[i]
        # y = df_traces[neuron_name][x]
        line.set_xdata(x)
        return line,

    # Save the animation as a video
    anim = animation.FuncAnimation(fig, animate, frames=t1-t0, interval=20, blit=True)
    anim.save(fname, fps=7, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    print(f"Saved video of trace for neuron {neuron_name} to {fname}")

    return fname


