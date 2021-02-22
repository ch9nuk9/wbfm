from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import tifffile as tiff
from csbdeep.utils import Path, normalize
from stardist.models import StarDist2D


def segment_with_stardist(vol_path):
    # segments a volume (via path)
    # vars set
    # Stardist models: versatile showed best results

    model_list = ['2D_paper_dsb2018', '2D_versatile_fluo', '2D_demo']
    mdl = model_list[0]  # '2D_versatile_fluo'
    axis_norm = (0, 1)
    n_channel = 1

    # logging/redirecting stdout
    log_file = "./log_" + mdl + ".log"

    with open(log_file, 'w') as log:
        # load stardist model
        model = StarDist2D.from_pretrained(mdl)

        # open volume.tif
        with tiff.TiffFile(vol_path) as vol:
            # initialize output dimensions
            z = len(vol.pages)
            xy = vol.pages[1].shape
            output_file = np.zeros((z, *xy))

            # iterate over images to run stardist on single images
            for idx, page in enumerate(vol.pages):
                img = page.asarray()
                print(idx, img.shape, file=log)

                # normalizing images (stardist function)
                img = normalize(img, 1, 99.8, axis=axis_norm)

                # run the prediction
                labels, details = model.predict_instances(img)

                # save labels in 3D array for output
                output_file[idx] = labels

    return output_file
