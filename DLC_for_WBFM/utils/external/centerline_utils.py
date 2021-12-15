import numpy as np
from skimage import transform


def get_scale_fluo_to_behavior():
    # pixel sizes in um
    img1_px_size = 0.325
    img2_px_size = 2.4
    # img2_px_size = 2.3
    scale = img1_px_size / img2_px_size
    # print('scale is :', scale)
    return scale


def get_rotation_fluo_to_behavior():
    # return np.pi / 2.05
    return np.pi / 1.85


def get_translation_fluo_to_behavior():
    # return 430, 310  # (680,1000)
    return 430, 330  # (680,1000)


def get_full_transformation_fluo_to_behavior():
    # Example application:
    # tf_img2 = transform.warp(img2, tform2.inverse, output_shape=img1.shape)
    # Where img1 is the behavior and img2 is the fluorescence

    scale2 = get_scale_fluo_to_behavior()
    rotation2 = get_rotation_fluo_to_behavior()
    translation2 = get_translation_fluo_to_behavior()

    # transformation2
    tform2 = transform.SimilarityTransform(
        scale=scale2,
        rotation=rotation2,
        translation=translation2)

    return tform2


def get_full_transformation_behavior_to_fluo():
    # Inverse of get_full_transformation_fluo_to_behavior
    # TOCHECK

    scale2 = 1 / get_scale_fluo_to_behavior()
    rotation2 = - get_rotation_fluo_to_behavior()
    translation2 = [-t for t in get_translation_fluo_to_behavior()]

    # transformation2
    tform2 = transform.SimilarityTransform(
        scale=scale2,
        rotation=rotation2,
        translation=translation2)

    return tform2
