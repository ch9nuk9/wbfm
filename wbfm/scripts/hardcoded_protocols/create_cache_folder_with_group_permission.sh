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

# Loop through subfolders
find "$1" -type d -exec mkdir -p {}/.cache \;

# Set write group permission on only the .cache folders
find "$1" -type d -name .cache -exec chmod g+w {} \;
