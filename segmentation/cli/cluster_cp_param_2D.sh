# This is the main shell script, which will create jobs for parameter testing of cellpose segmentation.
# Run this script in the folder, which should contain the output. IT will create subfolders within that folder.
# The python script will find the training data

# initialze some things
echo 'Starting the parameter test'

LOG_DIR="$(pwd)/log"

#it will create a batch job with number of arrays=ZBNUMDIRS and start array_job.sh

for px in `seq 5 1 10`;
do
    for flow in `seq 0.4 0.05 0.8`;
    do

        # create array jobs here with $d/$i
        # change '.' to ',' for folder names
        PX=$(sed 's/\./\,/' <<< $px)
        FLOW=$(sed 's/\./\,/' <<< $flow)
        DIR="$(pwd)/pixels-$PX-flow-$FLOW"
        mkdir -p $DIR
        cd $DIR
        
        # options taken out: -J $(printf cp-param-test-%s ${DIR})
        sbatch  --mem-per-cpu=20G --qos=short --wrap="conda activate cellpose; python /users/niklas.khoss/segmentation-git/segmentation-git/cp_param_test_2D.py $px $flow"
        cd ..
        
    done
done

