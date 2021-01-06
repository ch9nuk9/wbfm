import deeplabcut
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV as vp


def make_labeled_video_custom_annotations(dlc_config,
                                          video_fname,
                                          df):
    """
    Wrapper around the deeplabcut video creation functions to work with custom
    annotations
    """

    videooutname = video_fname.replace('.avi', '_labeled.mp4')
    codec="mp4v"
    clip = vp(fname=video_fname, sname=videooutname, codec=codec)
    cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)

    displayedbodyparts="all"
    bodyparts = deeplabcut.auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
            cfg, displayedbodyparts
        )
    labeled_bpts = [
        bp
        for bp in df.columns.get_level_values("bodyparts").unique()
        if bp in bodyparts
    ]

    trailpoints = 0

    cropping = False
    [x1, x2, y1, y2] = [0,0,0,0]

    bodyparts2connect = False
    draw_skeleton = False
    skeleton_color = None
    displaycropped = False
    color_by = "bodypart"

    # Actual function call
    CreateVideo(clip,
                df,
                cfg["pcutoff"],
                cfg["dotsize"],
                cfg["colormap"],
                labeled_bpts,
                trailpoints,
                cropping,
                x1,
                x2,
                y1,
                y2,
                bodyparts2connect,
                skeleton_color,
                draw_skeleton,
                displaycropped,
                color_by
            )
