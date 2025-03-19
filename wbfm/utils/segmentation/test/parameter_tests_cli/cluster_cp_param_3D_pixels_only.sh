# This is the main shell script, which will create jobs for parameter testing of cellpose segmentation.
# Run this script in the folder, which should contain the output. IT will create subfolders within that folder.
# The python script will find the training data

# initialze
echo 'Starting the parameter test'

# debug commandline argument:
# $debug=${1:-false}

# write a metadata.csv file containing the parameter name and values (per row), which are being tested

# check, if 'metadata.csv' file exists already. If not, create it
if [[ ! -f "./metadata.csv" ]];
then
    touch "./metadata.csv";
else
    printf "Metadata file exists already! Please rename or remove \"metadata.csv\"."
    echo "EXITING NOW"
    exit 1
fi
    
# create lists of numerical parameters for testing in python script

LIST1=($(seq 5 1 12))
printf "parameter1 " >> metadata.csv
echo ${LIST1[@]} >> metadata.csv


LOG_DIR="$(pwd)/log"

# TODO: create a list of parameter values and loop over the list later on

# TODO: log the parameter lists (one per row) in a log file (.csv)
# scheme = name of param, value1, value2
    

# it iterates over the wanted parameters (px = diameter in pixels, flow = flow threshold)


for px in ${LIST1[@]};
do

    # create array jobs here with $d/$i

    # change '.' to ',' for folder names
    PX=$(sed 's/\./\,/' <<< $px)

    DIR="$(pwd)/pixels-$PX-3D"

    mkdir -p $DIR
    cd $DIR
    # options taken out: -J $(printf cp-param-test-%s ${DIR})
    sbatch --mem-per-cpu=48G --qos=medium --time=08:00:00 --wrap="conda activate cellpose; python /users/niklas.khoss/segmentation-git/segmentation/cp_param_test_3D_pixels_only.py $px"
    cd ..
        
done

