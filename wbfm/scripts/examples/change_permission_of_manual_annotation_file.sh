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

# Use find to discover subfolders and apply chmod
find "$parent_dir" -maxdepth 1 -mindepth 1 -type d -name -wholename "*/$folder_path" -exec chmod -R g+w,o+rw "{}" \;
