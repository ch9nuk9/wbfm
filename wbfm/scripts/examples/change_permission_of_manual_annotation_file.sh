#!/bin/bash

# Get the target folder
while getopts s: flag
do
    case "${flag}" in
        s) parent_dir=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Define the file path relative to each subfolder
folder_path="3-tracking/manual_annotation"

# Iterate through top-level subfolders and apply permissions
for subfolder in "$parent_dir"/*/; do
    target_dir="$subfolder$folder_path"
    if [ -d "$target_dir" ]; then
        chmod -R g+w,o+rw "$target_dir"
        echo "Changed permissions for $target_dir"
    fi
done