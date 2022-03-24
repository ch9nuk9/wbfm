import pandas as pd
from matplotlib import pyplot as plt
from sacred import Experiment

from DLC_for_WBFM.utils.nn_utils.model_image_classifier import NullModel
from DLC_for_WBFM.utils.nn_utils.utils_testing import plot_accuracy, test_open_set_tracking
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.utils_filenames import add_name_suffix
ex = Experiment()


@ex.automain
def main():
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
    num_frames = None
    raw_frames = project_data.raw_frames
    if num_frames is None:
        frames_to_test = raw_frames
    else:
        frames_to_test = [raw_frames[t] for t in range(num_frames)]

    ## Baseline
    model = NullModel()

    correct_per_class, total_per_class, name_mapping, accuracy_correct_per_class, accuracy_incorrect_per_class, mean_acc = \
        test_open_set_tracking(project_data, model, neurons_that_are_finished, all_frames=frames_to_test)

    plot_accuracy(correct_per_class, total_per_class)
    plt.xticks(rotation=90)
    plt.title(f"Accuracy={mean_acc}")

    suffix = '-baseline'
    fname = 'plots/classifier_accuracy.png'
    fname = add_name_suffix(fname, suffix=suffix)
    plt.savefig(fname)

    ##
