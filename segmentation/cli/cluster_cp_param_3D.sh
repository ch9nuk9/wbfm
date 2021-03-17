# This is the main shell script, which will create jobs for parameter testing of cellpose segmentation.
# Run this script in the folder, which should contain the output. IT will create subfolders within that folder.
# The python script will find the training data

# initialze some things
echo 'Starting the parameter test'
       

LOG_DIR="$(pwd)/log"

# it iterates over the wanted parameters (px = diameter in pixels, flow = flow threshold)

for px in `seq 5 1 12`;
do
    for flow in `seq 0.4 0.05 0.8`;
    do

        # create array jobs here with $d/$i
        
        # change '.' to ',' for folder names
        PX=$(sed 's/\./\,/' <<< $px)
        FLOW=$(sed 's/\./\,/' <<< $flow)
        
        DIR="$(pwd)/pixels-$PX-flow-$FLOW-3D"
        mkdir -p $DIR
        cd $DIR
        # options taken out: -J $(printf cp-param-test-%s ${DIR})
        sbatch --mem-per-cpu=48G --qos=medium --wrap="conda activate cellpose; python /users/niklas.khoss/segmentation-git/segmentation-git/cp_param_test_3D.py $px $flow"
        cd ..
        
    done
done

