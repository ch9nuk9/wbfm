#!/bin/bash

# Just go to hardcoded folders and use pip to install them
cd /home/charles/Current_work/repos/dlc_for_wbfm
pip install -e .

cd /home/charles/Current_work/repos/barlow_track/
pip install -e .

cd /home/charles/PycharmProjects/imutils
pip install -e .

cd /home/charles/PycharmProjects/centerline_behavior_annotation
pip install -e .

