#!/bin/bash

CMD="python3 -m pytest /dlc_for_wbfm/DLC_for_WBFM/tests/test_0-installation.py"
# Run tests of the installation
sudo docker run wbfm:gpu "${CMD}"
