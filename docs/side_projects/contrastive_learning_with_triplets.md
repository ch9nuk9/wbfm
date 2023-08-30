# Docs for side project: using contrastive learning

Basic steps:
1. Generate training data (triplets) from a project with ground truth
2. Train a neural network
3. Test using clustering

## Generate training data

See the function save_training_data in this [file](../../wbfm/utils/barlow_project/utils/data_loading.py)

This currently uses one of my internal tracklet-specific dataframes, but should be refactored to use ground truth tracks

## Train a neural network

See [example training script](../../wbfm/utils/barlow_project/scripts/train_triplet_image_space.py)

This uses classes from [here](../../wbfm/utils/barlow_project/utils/data_loading.py), especially the NeuronTripletDataset

Several choices are completely unoptimized, including:
1. Metric
2. Criterion
3. How triplets are chosen
4. Network architecture

## Test final performance

Main comparison is between two dataframes using the function calculate_accuracy_from_dataframes, [here](../../wbfm/utils/performance/comparing_ground_truth.py)
