# DLC_for_WBFM

This repository has all my functions for combining DeepLabCut with Whole Brain Freely Moving Imaging

## Installation

The workflow for this uses three different environments that are in flux, and will be consolidated once the project finishes. Requirements to build these environments are in the 'conda-environments' folder. Currently these are:

1. DLC-preprocessing.yaml
   1. Designed for preprocessing, in particular custom annotation and video conversion functions.
2. DLC-CPU.yaml
   1. This is the main DeepLabCut environment.
   2. Could also be installed on GPU, using DLC-GPU.yaml
3. DLC-postprocessing.yaml
   1. Mostly focused on getting the traces using location information
   2. Uses 'dNMF' which currently must be installed locally.


## Usage

See example-immobilized-workflow/ or example-WBFM-workflow/ for notebooks describing the workflow of your dataset.

## Explanation of directories

### DLC_for_WBFM

Contains the main utilities for preprocessing and postprocessing. DeepLabCut itself is not included, and should be installed separately

### docs

Contains documentation, in particular flow charts describing a normal workflow and the planned upgrades for this package

### example-immobilized-workflow

Contains notebooks for full workflows on immobilized data. This data does not require as much pre- and post-processing as freely moving data, but is organized in three notebooks:

1. preprocessing
2. Workflow (DeepLabCut training)
3. postprocessing

### example-visualizations

Largely older notebooks

### example-WBFM-workflow

Example notebooks for the Freely Moving datasets. Note that this is still actively changing, and is organized in 3 notebooks:

1. preprocessing
2. Workflow (DeepLabCut training)
3. postprocessing

### Tutorials

Notebooks on basic tasks, mostly video conversion
