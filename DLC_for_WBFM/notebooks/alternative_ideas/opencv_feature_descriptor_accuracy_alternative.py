import cv2
import pandas as pd
import sacred
from matplotlib import pyplot as plt
from sacred import Experiment

from wbfm.utils.neuron_matching.feature_pipeline import calculate_frame_objects_full_video
from wbfm.utils.nn_utils.model_image_classifier import NullModel
from wbfm.utils.nn_utils.utils_testing import plot_accuracy, test_open_set_tracking
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_filenames import add_name_suffix
ex = Experiment()
ex.add_config(encoder_type='baseline', DEBUG=False)


def test_feature_encoder(encoder_type='baseline', encoder_kwargs=None, DEBUG=False):
    if encoder_kwargs is None:
        encoder_kwargs = {}

    fname = "/home/charles/dlc_stacks/worm1_for_students/project_config-workstation.yaml"
    project_data = ProjectData.load_final_project_data_from_config(fname)
    track_cfg = project_data.project_config.get_tracking_config()
    fname = track_cfg.resolve_relative_path("manual_annotation/manual_tracking.csv", prepend_subfolder=True)
    df_manual_tracking = pd.read_csv(fname)
    neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['Finished?']]['Neuron ID'])
    # neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['first 1300 frames']]['Neuron ID'])
    num_finished = len(neurons_that_are_finished)
    print(f"Found {num_finished}/{len(df_manual_tracking)} finished neurons")
    ## Setup
    tracker_opt = {}

    training_config = project_data.project_config.get_training_config()
    metadata_fname = training_config.config['tracker_params']['external_detections']
    external_detections = training_config.resolve_relative_path(metadata_fname)
    if DEBUG:
        num_frames = 11
    else:
        num_frames = project_data.num_frames
    start_volume = 0
    end_volume = num_frames
    video_fname = project_data.project_config.config['preprocessed_red']
    z_depth_neuron_encoding = 3
    # Define new encoder

    if encoder_type == 'baseline':
        base_2d_encoder = cv2.xfeatures2d.VGG_create(**encoder_kwargs)
    elif encoder_type == 'vgg_different_defaults':
        encoder_kwargs.update(dict(img_normalize=False, use_scale_orientation=False))
        base_2d_encoder = cv2.xfeatures2d.VGG_create(**encoder_kwargs)
    elif encoder_type == 'latch':
        base_2d_encoder = cv2.xfeatures2d.LATCH_create(**encoder_kwargs)
    elif encoder_type == 'latch_different_defaults':
        encoder_kwargs.update(dict(sigma=1.4))
        base_2d_encoder = cv2.xfeatures2d.LATCH_create(**encoder_kwargs)
    elif encoder_type == 'daisy':
        base_2d_encoder = cv2.xfeatures2d.DAISY_create(**encoder_kwargs)
    elif encoder_type == 'daisy_different_defaults':
        encoder_kwargs.update(dict(interpolation=False, q_theta=4, q_hist=4))
        base_2d_encoder = cv2.xfeatures2d.DAISY_create(**encoder_kwargs)
    elif encoder_type == 'akaze':
        base_2d_encoder = cv2.AKAZE_create(**encoder_kwargs)
    elif encoder_type == 'freak':
        base_2d_encoder = cv2.xfeatures2d.FREAK_create(**encoder_kwargs)
    elif encoder_type == 'freak_different_defaults':
        encoder_kwargs.update(dict(scaleNormalized=False))
        base_2d_encoder = cv2.xfeatures2d.FREAK_create(**encoder_kwargs)
    elif encoder_type == 'orb':
        encoder_kwargs.update(dict(edgeThreshold=0))
        base_2d_encoder = cv2.ORB_create(**encoder_kwargs)

        tracker_opt = dict(cdist_p=0)
    elif encoder_type == 'orb_different_defaults':
        encoder_kwargs.update(dict(edgeThreshold=0, patchSize=71, scaleFactor=1.5))
        base_2d_encoder = cv2.ORB_create(**encoder_kwargs)
        tracker_opt = dict(cdist_p=0)

    else:
        raise NotImplementedError

    encoder_opt = dict(base_2d_encoder=base_2d_encoder)
    frames_to_test = calculate_frame_objects_full_video(external_detections, start_volume, end_volume, video_fname,
                                                        z_depth_neuron_encoding, encoder_opt, max_workers=1)

    ## Use as tracker
    model = NullModel()
    correct_per_class, total_per_class, name_mapping, accuracy_correct_per_class, accuracy_incorrect_per_class, mean_acc, _ = \
        test_open_set_tracking(project_data, neurons_that_are_finished, model=model, all_frames=frames_to_test, tracker_opt=tracker_opt)
    plot_accuracy(correct_per_class, total_per_class)
    plt.xticks(rotation=90)
    plt.title(f"Accuracy={mean_acc}")
    suffix = f'-{encoder_type}'
    fname = f'plots/classifier_accuracy_{encoder_kwargs}.png'
    fname = fname.replace('{', '').replace('}', '')
    fname = add_name_suffix(fname, suffix=suffix)
    plt.savefig(fname)
    ##


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    test_feature_encoder(_config['encoder_type'], _config['DEBUG'])
