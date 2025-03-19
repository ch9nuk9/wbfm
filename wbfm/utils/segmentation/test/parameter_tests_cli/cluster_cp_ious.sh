# job array for calculatign the IoUs of the cellpose (3D) results

# This is the main shell script, which will create jobs to calculate the IoUs of the cellpose results.
# Run this script in the parentfolder of the cellpose 3D runs, i.e. /groups/zimmer/shared_projects/wbfm/cellpose_parameter_test/3D_tests

# initialze some things
echo 'Starting the IoU calculations'
LOG_DIR="$(pwd)/log"

# iterate over subdirectories
for dir in ./*/; do
    [ -d "${dir}" ] || continue
#     echo $dir
    cd $dir
    # gets path of cellpose result (a mask; .npy)
    MASK_PATH="$(find ~+ -name "np_masks*npy")"
    # call python wrapper script with argument (alternatively, find the mask within the python script)
    sbatch --mem-per-cpu=16G --qos=short --wrap="conda activate cellpose;
    python /users/niklas.khoss/segmentation-git/segmentation-git/wrapper_iou.py $MASK_PATH"
    cd ..
done