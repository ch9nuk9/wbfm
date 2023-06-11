import dask.array
import napari
import tifffile
import zarr

from wbfm.gui.utils.utils_gui import change_viewer_time_point
from wbfm.gui.utils.utils_gui_matplot import PlotQWidget
from wbfm.utils.projects.finished_project_data import ProjectData


def main():
    # Load project
    fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
    project_data = ProjectData.load_final_project_data_from_config(fname)

    # Load specific data
    df_kymo = project_data.worm_posture_class.curvature(fluorescence_fps=True)
    video_fname = project_data.worm_posture_class.behavior_video_avi_fname()
    video_fname = video_fname.with_suffix('.btf')
    store = tifffile.imread(video_fname, aszarr=True)
    video_zarr = zarr.open(store, mode='r')
    # Subset video to be the same fps as the fluorescence
    video_zarr = video_zarr[::project_data.worm_posture_class.frames_per_volume, :, :]
    video = video_zarr

    # dask_chunk = list(video_zarr.chunks).copy()
    # dask_chunk[0] = 1000
    # video = dask.array.from_zarr(video_zarr, chunks=dask_chunk)

    # Main viewer
    viewer = napari.view_image(video)

    # Kymograph subplot
    mpl_widget = PlotQWidget()
    static_ax = mpl_widget.canvas.fig.subplots()
    static_ax.imshow(df_kymo.T, aspect=1, vmin=-0.05, vmax=0.05, cmap='RdBu')
    viewer.window.add_dock_widget(mpl_widget, area='bottom')

    # Callback: click on the kymograph to change the viewer time
    def on_subplot_click(event):
        t = event.xdata
        change_viewer_time_point(viewer, t_target=t)
    mpl_widget.canvas.mpl_connect('button_press_event', on_subplot_click)

    # Callback: when the time is changed, update a vertical line on the kymograph
    def get_time_line_options():
        return dict(x=viewer.dims.current_step[0], color='black')

    time_line = static_ax.axvline(**get_time_line_options())

    @viewer.dims.events.current_step.connect
    def update_time_line(event):
        time_options = get_time_line_options()
        time_line.set_xdata(time_options['x'])
        time_line.set_color(time_options['color'])
        mpl_widget.draw()

    # Start gui loop
    napari.run()


if __name__ == "__main__":
    main()
