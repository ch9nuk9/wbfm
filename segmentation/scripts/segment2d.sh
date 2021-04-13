#!C:\Users\charles.fieseler\AppData\Local\Programs\Git\usr\bin\bash.exe

DAT_PATH="D:\More-stabilized-wbfm\test2020-10-22_16-15-20_test4-channel-0-pco_camera1\test2020-10-22_16-15-20_test4-channel-0-pco_camera1bigtiff.btf"
COMMAND="scripts/segment2d.py"
# echo $(pwd)
python $COMMAND with video_path=$DAT_PATH dataset_params.start_volume=100
