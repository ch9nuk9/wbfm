#!/bin/bash

# Get the target folder
while getopts t: flag
do
    case "${flag}" in
        t) parent_dir=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Define the file path relative to each subfolder
file_path="3-tracking/manual_annotation/manual_annotation.xlsx"

# Use find to discover subfolders and apply chmod
find "$parent_dir" -maxdepth 1 -mindepth 1 -type d -exec chmod g+w {}/"$file_path" \;
