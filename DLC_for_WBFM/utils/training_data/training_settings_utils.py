import deeplabcut
import os
import pathlib

##
## Helper functions for good defaults
##


# def update_pose_cfg(path_config_file):
#     """
#     DEPRECATED
#     Updates the pose_cfg file with more appropriate defaults for WBFM
#
#     Specifically, adds:
#         Data augmentation
#
#         Faster learning rate, then slower
#
#     By default updates only iteration-0
#     """
#
#     config_file = pathlib.Path(path_config_file).resolve()
#
#     ## Get the config file
#     project_folder = os.path.dirname(path_config_file)
#     # ENHANCE: access different iterations
#     iteration_folder = os.path.join(project_folder, 'dlc-models', 'iteration-0')
#     # ENHANCE: access different shuffles
#     this_shuffle = os.listdir(iteration_folder)[0]
#     training_folder = os.path.join(iteration_folder, this_shuffle, 'train')
#
#     training_config_fname = os.path.join(training_folder, 'pose_cfg.yaml')
#
#     ## Dictionary of updates
#     updates = {"rotation": 180, #Tensorpack
#                "rotate_max_deg_abs": 180, #img_aug
#                "multi_step":[
#                [0.02, 1000],
#                [0.005, 5000],
#                [0.001, 20000]
#                ]}
#
#     deeplabcut.auxiliaryfunctions.edit_config(training_config_fname, updates)
#
#     print("Finished updating training config file")
