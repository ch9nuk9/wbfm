for dir in ./*/; do
    [ -d "${dir}" ] || continue
    cd $dir
    # gets path of cellpose result (a mask; .npy)
    MASK_PATH="$(find ~+ -name "np_masks*npy")"
    echo $MASK_PATH
    cd ..
done