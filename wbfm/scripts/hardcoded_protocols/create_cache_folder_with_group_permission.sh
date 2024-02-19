#!/bin/bash

# For a given folder, loop through subfolders and create a .cache folder with group permission
# Usage: ./create_cache_folder_with_group_permission.sh /path/to/folder

# Check if folder is provided
if [ -z "$1" ]; then
    echo "Please provide a folder"
    exit 1
fi

# Check if folder exists
if [ ! -d "$1" ]; then
    echo "Folder does not exist"
    exit 1
fi

# Loop through subfolders, and print the .cache folder
for dir in $1/*; do
    if [ -d "$dir" ]; then
        mkdir -pv "$dir/.cache"
        # Change permissions to allow group write
        chmod g+w "$dir/.cache"
    fi
done
