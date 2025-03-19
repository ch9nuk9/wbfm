#!/usr/bin/env python
# coding: utf-8

# **In case of problems or questions, please first check the list of [Frequently Asked Questions (FAQ)](https://stardist.net/docs/faq.html).**
# 
# Please shutdown all other training/prediction notebooks before running this notebook (as those might occupy the GPU memory otherwise).

# In[1]:


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from csbdeep.utils import  normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching_dataset
from stardist.models import Config3D, StarDist3D
import zarr
import os

np.random.seed(42)
lbl_cmap = random_label_cmap()

print(gputools_available())

# # Prepare the Data
# 
# We assume that data has already been downloaded via notebook [1_data.ipynb](1_data.ipynb).  
# 
# <div class="alert alert-block alert-info">
# Training data (for input `X` with associated label masks `Y`) can be provided via lists of numpy arrays, where each image can have a different size. Alternatively, a single numpy array can also be used if all images have the same size.  
# Input images can either be three-dimensional (single-channel) or four-dimensional (multi-channel) arrays, where the channel axis comes last. Label images need to be integer-valued.
# </div>


folder = '/home/charles/Current_work/ground_truth/students_segmentation_annotations/'

# Open all masks and data, then concat
all_suffixes = ['worm_5', 'worm_4', 'imm_worm_6', 'imm_worm_4']

all_y = []
all_x = []
for suffix in all_suffixes:
    fname = os.path.join(folder, f'masks_{suffix}.zarr')
    all_y.append(zarr.open(fname))

    fname = os.path.join(folder, f'red_{suffix}.zarr')
    all_x.append(zarr.open(fname))

# Also add Lukas' annotations
fname = os.path.join('/home/charles/Current_work/ground_truth/lukas_segmentation_annotations/masks_cropped.zarr/')
all_y.append(zarr.open(fname)[:, 1:, ...])

fname = os.path.join('/home/charles/Current_work/ground_truth/lukas_segmentation_annotations/red_cropped.zarr/')
all_x.append(zarr.open(fname)[:, 1:, ...])

# Based on manual check

X_raw = np.vstack(all_x)
Y_raw = np.vstack(all_y)
print(X_raw.shape)

# This should be fixed at the data saving side
# ind_to_remove = [1, 11, 12, 13, 20, 21, 35, 36, 37, 38, 39]
# ind_to_keep = [i for i in range(X_raw.shape[0]) if i not in ind_to_remove]
# X_raw = X_raw[ind_to_keep, ...]
# Y_raw = Y_raw[ind_to_keep, ...]
# print(X_raw.shape)

num_frames = X_raw.shape[0]
# n_channel = 1 if X_raw[0].ndim == 3 else X_raw[0].shape[-1]
n_channel = 1


# In[6]:


# Visualize in 3d
import napari

v = napari.Viewer(ndisplay=3)
v.add_image(X_raw)
# v.add_labels(Y_raw)


# Normalize images and fill small label holes.

# In[13]:


axis_norm = (0,1,2)   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
    sys.stdout.flush()

# NEW: translate from video to list of volumes
# X = [normalize(np.array(X[i,...],dtype='uint8'),1,99.8,axis=axis_norm) for i in tqdm(ind_vec)]
X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_raw)]
# BROKEN LINE
# Y = [fill_label_holes(np.array(Y[i,...],dtype='uint8')) for i in tqdm(ind_vec)]
Y = [fill_label_holes(np.array(y,dtype='uint8')) for y in tqdm(Y_raw)]


# Split into train and validation datasets.

# In[14]:


assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))


# Training data consists of pairs of input image and label instances.

# In[15]:


def plot_img_label(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None, **kwargs):
    if z is None:
        z = img.shape[0] // 2    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


# In[16]:


i = 0
img, lbl = X[i], Y[i]
assert img.ndim in (3,4)
img = img if img.ndim==3 else img[...,:3]
plot_img_label(img,lbl)
None;


# # Configuration
# 
# A `StarDist3D` model is specified via a `Config3D` object.

# In[17]:


# print(Config3D.__doc__)


# In[18]:


extents = calculate_extents(Y)
anisotropy = tuple(np.max(extents) / extents)
print('empirical anisotropy of labeled objects = %s' % str(anisotropy))


# In[19]:


# 96 is a good default choice (see 1_data.ipynb)
n_rays = 96

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

conf = Config3D (
    rays             = rays,
    grid             = grid,
    anisotropy       = anisotropy,
    use_gpu          = use_gpu,
    n_channel_in     = n_channel,
    # adjust for your data below (make patch size as large as possible)
    train_patch_size = (8,96,96),
    train_batch_size = 2,
)
# print(conf)
# vars(conf)


# In[20]:


if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)
    # alternatively, try this:
    # limit_gpu_memory(None, allow_growth=True)


# **Note:** The trained `StarDist3D` model will *not* predict completed shapes for partially visible objects at the image boundary.

# In[30]:


# model = StarDist3D(conf, name='Students_and_Lukas3d_zarr', basedir='models')
# Start from previous model
model = StarDist3D(None, name='Students_and_Lukas3d_zarr', basedir='models')


# Check if the neural network has a large enough field of view to see up to the boundary of most objects.

# In[31]:


median_size = calculate_extents(Y, np.median)
fov = np.array(model._axes_tile_overlap('ZYX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")


# # Data Augmentation

# You can define a function/callable that applies augmentation to each batch of the data generator.  
# We here use an `augmenter` that applies random rotations, flips, and intensity changes, which are typically sensible for (3D) microscopy images (but you can disable augmentation by setting `augmenter = None`).

# In[32]:


def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y


# In[33]:


# plot some augmented examples
img, lbl = X[0],Y[0]
plot_img_label(img, lbl)
for _ in range(3):
    img_aug, lbl_aug = augmenter(img,lbl)
    plot_img_label(img_aug, lbl_aug, img_title="image augmented (XY slice)", lbl_title="label augmented (XY slice)")


# # Training

# We recommend to monitor the progress during training with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). You can start it in the shell from the current working directory like this:
# 
#     $ tensorboard --logdir=.
# 
# Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.
# 

# In[ ]:


quick_demo = False

if quick_demo:
    print (
        "NOTE: This is only for a quick demonstration!\n"
        "      Please set the variable 'quick_demo = False' for proper (long) training.",
        file=sys.stderr, flush=True
    )
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                epochs=2, steps_per_epoch=5)

    print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
    model = StarDist3D.from_pretrained('3D_demo')
else:
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=1000)
None;


# # Threshold optimization

# While the default values for the probability and non-maximum suppression thresholds already yield good results in many cases, we still recommend to adapt the thresholds to your data. The optimized threshold values are saved to disk and will be automatically loaded with the model.

# In[ ]:


# Reload
model = StarDist3D(None, name='Students_and_Lukas3d_zarr', basedir='models')


# In[ ]:


if quick_demo:
    # only use a single validation image for demo
    model.optimize_thresholds(X_val[:1], Y_val[:1])
else:
    model.optimize_thresholds(X_val, Y_val)


# # Evaluation and Detection Performance

# Besides the losses and metrics during training, we can also quantitatively evaluate the actual detection/segmentation performance on the validation data by considering objects in the ground truth to be correctly matched if there are predicted objects with overlap (here [intersection over union (IoU)](https://en.wikipedia.org/wiki/Jaccard_index)) beyond a chosen IoU threshold $\tau$.
# 
# The corresponding matching statistics (average overlap, accuracy, recall, precision, etc.) are typically of greater practical relevance than the losses/metrics computed during training (but harder to formulate as a loss function). 
# The value of $\tau$ can be between 0 (even slightly overlapping objects count as correctly predicted) and 1 (only pixel-perfectly overlapping objects count) and which $\tau$ to use depends on the needed segmentation precision/application.
# 
# Please see `help(matching)` for definitions of the abbreviations used in the evaluation below and see the Wikipedia page on [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) for further details.

# In[ ]:


# help(matching)


# First predict the labels for all validation images:

# In[ ]:


Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]


# Plot a GT/prediction example  

# In[ ]:


plot_img_label(X_val[0],Y_val[0], lbl_title="label GT (XY slice)")
plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred (XY slice)")


# Choose several IoU thresholds $\tau$ that might be of interest and for each compute matching statistics for the validation data.

# In[ ]:


taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]


# Example: Print all available matching statistics for $\tau=0.7$

# In[ ]:


stats[taus.index(0.7)]


# Plot the matching statistics and the number of true/false positives/negatives as a function of the IoU threshold $\tau$. 

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
    ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax1.set_xlabel(r'IoU threshold $\tau$')
ax1.set_ylabel('Metric value')
ax1.grid()
ax1.legend()

for m in ('fp', 'tp', 'fn'):
    ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax2.set_xlabel(r'IoU threshold $\tau$')
ax2.set_ylabel('Number #')
ax2.grid()
ax2.legend();


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




