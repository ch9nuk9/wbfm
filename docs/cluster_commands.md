# A quick guide for cluster commands

## Preprocessing

### Convert a folder of .stk to one bigtiff

What you need first:
Folder of .stk files (from a microscope)
Output directory (can be same)
Terminal on the cluster (e.g. ssh)


Command:
''''
srun -o log.log -e errors.log --mem=32G bfconvert 27082020_trial2_dual1_HEAD_t1.stk 27082020_trial2_dual1_HEAD.ome.btf &
''''

Explanation:

1. srun => the command to run a batch job on the cluster
2. -o log.log => outputs the text to a file instead of the screen
3. -e errors.log => outputs the errors to a file instead of the screen
4. --mem=32G => increases the RAM of the job; default is 4G
5. bfconvert => the actual script to run
6. 27082020_trial2_dual1_HEAD_t1.stk => the input file, which is ONE of the files in the folder. The metadata should allow the script to find the other files
7. 27082020_trial2_dual1_HEAD.ome.btf => the output files
8. & => Runs in the background. If you do not do this, then your job will quit if you close the terminal/connection

During running:
This command shows you all your jobs:
''''
squeue --me
''''
Which will output something like this:

JOBID PARTITION  |   NAME |    USER |ST  |     TIME | NODES |NODELIST(REASON)
|---|---|---|---|---|---|---|
10253058 |        c| bfconver |charles. | R  |     9:02   |   1| clip-c2-23
10253015  |       c| bfconver |charles. | R |      9:21   |   1| clip-c2-37
