
# Installation for developing

### Install Anaconda

https://www.anaconda.com/products/individual

There are additional details in the computational protocols repository:
https://github.com/Zimmer-lab/protocols/tree/master/computational/zimmer_lab_code_pipeline

### Get the code

Download or clone these repositories:
1. wbfm (this repo): https://github.com/Zimmer-lab/wbfm
2. Segmentation: https://github.com/Zimmer-lab/segmentation
3. imutils (Lukas' data reader): https://github.com/Zimmer-lab/imutils

### Install the environments

#### Pre-installed environments

Note: there are pre-installed environments living on the cluster, at:
/lisc/scratch/zimmer/.conda/envs/wbfm

They can be activated using:
```commandline
conda activate /lisc/scratch/zimmer/.conda/envs/wbfm
```

#### Installing new environments

Note: if you are just using the GUI, then you can use a simplified environment.
Detailed instructions can be found in the [README](wbfm/gui/README.md) file under the gui section
For running the full pipeline you need the environment found here:

1. conda-environments/wbfm.yaml

This installs the public packages, now we need to install our local libraries.
Do `conda activate wbfm` (or whatever your name is) and install the local code in the following way:

1. cd to the segmentation repository
2. run: `pip install -e .`
3. Repeat steps 1-2 for the other repositories, wbfm and imutils

#### Summary of installations

You will install 4 "things": 1 environment and 3 custom packages
