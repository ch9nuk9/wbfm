
# Installation for developing

### Install Anaconda

https://www.anaconda.com/products/individual

There are additional details in the computational protocols repository:
https://github.com/Zimmer-lab/protocols/tree/master/computational/zimmer_lab_code_pipeline

### Get the code

Download or clone these repositories:
1. wbfm (this repo): https://github.com/Zimmer-lab/wbfm
2. Segmentation: https://github.com/Zimmer-lab/segmentation
3. fDNC (leifer paper): https://github.com/Charles-Fieseler-Vienna/fDNC_Neuron_ID

### Install the environments

#### Pre-installed environments

Note: there are pre-installed environments living on the cluster, at:
/scratch/zimmer/.conda/envs/segmentation
/scratch/zimmer/.conda/envs/wbfm

They can be activated using:
```commandline
conda activate/lisc/scratch/zimmer/.conda/envs/wbfm
```

#### Installing new environments

Note: if you are just using the GUI, then you can use a simplified environment.
Detailed instructions can be found in the [README](wbfm/gui/README.md) file under the gui section

For this project, two separate environments are currently needed.
This is due to versioning interactions between opencv and tensorflow within deeplabcut and stardist. 
Both environments can be found in the folder conda-environments/

1. segmentation.yaml
2. wbfm.yaml

This installs the public packages, now we need to install our custom code.
Do `conda activate segmentation` and install the local code in the following way:

1. cd to the segmentation repository
2. run: `pip install -e .`
3. Repeat steps 1-2 for the other repository, wbfm
4. Repeat steps 1-3 for the other environment

#### Summary of installations

You will install 4 things: 2 environments and 3 custom packages
