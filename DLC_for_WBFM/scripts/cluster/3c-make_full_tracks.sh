#!/usr/bin/env bash

CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/3c-make_full_tracks.py"
project_path="$1"

python $CMD with project_path=$project_path
