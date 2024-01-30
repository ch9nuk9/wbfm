import glob
import os
from pathlib import Path

from wbfm.utils.projects.project_config_classes import ModularProjectConfig


configfile: "snakemake_config.yaml"

# Determine the project folder (the parent of the folder containing the Snakefile)
# NOTE: this is an undocumented feature, and may not work for other versions (this is 7.32)
project_dir = os.path.dirname(snakemake.workflow.workflow.basedir)
print("Detected project folder: ", project_dir)

# Load the folders needed for the behavioral part of the pipeline
cfg = ModularProjectConfig(project_dir)
raw_data_dir, output_behavior_dir = cfg.get_folders_for_behavior_pipeline()


def _run_helper(script_name, project_path):
    import importlib
    print("Running script: ", script_name)
    _module = importlib.import_module(f"wbfm.scripts.{script_name}")
    _module.ex.run(config_updates=dict(project_path=project_path))


def _cleanup_helper(output_path):
    """Uses the snakemake defined temporary function to clean up intermediate files, based on a flag"""
    if config['delete_intermediate_files']:
        return temporary(output_path)
    else:
        return output_path


#
# Snakemake for overall targets (either with or without behavior)
#

# By default, wbfm projects will run everything
rule traces_and_behavior:
    input:
        traces=expand("{dir}/{test}", test=config['output_4'], dir=project_dir),
        beh_figure= f"{output_behavior_dir}/behavioral_summary_figure.pdf"

# This is important for immobilized worms, which don't have behavior
rule traces:
    input:
        traces=expand("{dir}/{test}", test=config['output_4'], dir=project_dir)

rule behavior:
    input:
        beh_figure= f"{output_behavior_dir}/behavioral_summary_figure.pdf"

#
# Snakemake for traces
#

rule preprocessing:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
    output:
        expand("{dir}/{output}", output=config['output_0'], dir=project_dir),
    run:
        _run_helper(config['script_0'], str(input.cfg))

#
# Segmentation
#
rule segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        files=expand("{dir}/{input}", input=config['input_1'], dir=project_dir)
    output:
        metadata=expand("{dir}/{output}", output=config['output_1'], dir=project_dir),
        masks=directory(expand("{dir}/{output}", output=config['output_1_dir'], dir=project_dir))
    threads: 56
    run:
        _run_helper(config['script_1'], str(input.cfg))


#
# Tracklets
#
rule build_frame_objects:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        masks=ancient(expand("{dir}/{input}", input=config['output_1_dir'], dir=project_dir)),
        files=expand("{dir}/{input}", input=config['input_2a'], dir=project_dir)
    output:
        expand("{dir}/{output}", output=config['output_2a'], dir=project_dir)
    threads: 56
    run:
        _run_helper(config['script_2a'], str(input.cfg))


rule match_frame_pairs:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        masks=ancient(expand("{dir}/{input}", input=config['output_1_dir'], dir=project_dir)),
        files=expand("{dir}/{input}", input=config['input_2b'], dir=project_dir)
    output:
        expand("{dir}/{output}", output=config['output_2b'], dir=project_dir)
    threads: 56
    run:
        _run_helper(config['script_2b'], str(input.cfg))


rule postprocess_matches_to_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        files=expand("{dir}/{input}", input=config['input_2c'], dir=project_dir),
    output:
        expand("{dir}/{output}", output=config['output_2c'], dir=project_dir)
    threads: 8
    run:
        _run_helper(config['script_2c'], str(input.cfg))

#
# Tracking
#
rule tracking:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        files=expand("{dir}/{input}", input=config['input_3a'], dir=project_dir)
    output:
        expand("{dir}/{output}", output=config['output_3a'], dir=project_dir)
    threads: 48
    run:
        _run_helper(config['script_3a'], str(input.cfg))

rule combine_tracking_and_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        files=expand("{dir}/{input}", input=config['input_3b'], dir=project_dir)
    output:
        expand("{dir}/{output}", output=config['output_3b'], dir=project_dir)
    threads: 8
    run:
        _run_helper(config['script_3b'], str(input.cfg))

#
# Traces
#
rule extract_full_traces:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=project_dir),
        files=expand("{dir}/{input}", input=config['input_4'], dir=project_dir)
    output:
        expand("{dir}/{output}", output=config['output_4'], dir=project_dir)
    threads: 56
    run:
        _run_helper(config['script_4'], str(input.cfg))


#
# Behavioral analysis (kymographs)
#

# Find the specific folders to use
raw_data_subfolder = glob.glob(f"{raw_data_dir}/*Ch0-BH/")
print("Found raw data subfolder(s): ", raw_data_subfolder)
if len(raw_data_subfolder) == 1:
    raw_data_subfolder = raw_data_subfolder[0]
elif len(raw_data_subfolder) > 1:
    raise ValueError("There is more than one raw dataset")
else:
    raise ValueError("No raw data found")

# See if the .btf file has already been produced... unfortunately this is a modification of the raw data folder
# Assume that any .btf file is the correct one, IF it doesn't have 'AVG' in the name (refers to background subtraction)
btf_file = [f for f in os.listdir(raw_data_subfolder) if f.endswith(".btf") and 'AVG' not in f]
if len(btf_file) == 1:
    btf_file = btf_file[0]
    print(".btf file already produced: ", btf_file)
    btf_file = os.path.join(raw_data_subfolder, btf_file)
elif len(btf_file) > 1:
    raise ValueError(f"There is more than one .btf file in {raw_data_subfolder}")
else:
    # Then it will need to be produced
    btf_file = os.path.join(raw_data_subfolder, 'raw_stack.btf')
    print("WARNING: No .btf file found, will produce it in the raw data folder ", btf_file)

# Look for background image
background_img = glob.glob(f"{raw_data_dir}/../background/*background*BH*/*AVG*background*")
if len(background_img) == 1:
    background_img = background_img[0]
    background_img = str(Path(background_img).resolve()) # This is needed because the path is relative
    print("This is the background image: ", background_img)
elif len(background_img) > 1:
    raise ValueError(f"There is more than one background images in {raw_data_dir}/../background/")
else:
    raise ValueError(f"No background images found in {raw_data_dir}/../background/")

# Start snakemake

# TODO: this modifies the raw data folder... a decision should be made about this
rule ometiff2bigtiff:
    params:
        function = "ometiff2bigtiff"
    output:
        btf_file = {btf_file}
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            params.function,
            '-path', str(raw_data_subfolder),
            '-output_filename', str(output.btf_file),
        ])
    # shell:
    #     "python {input.script} {params.function} -path {wildcards.datasets} -output_filename {output.btf_file}"


rule subtract_background:
    input:
        raw_img  = {btf_file},
        background_img = background_img
    params:
        function = "stack_subtract_background",
        do_inverse = config["do_inverse"]
    output:
        background_subtracted_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            params.function,
            '-i', str(input.raw_img),
            '-o', str(output.background_subtracted_img),
            '-bg', str(input.background_img),
            '-invert', str(params.do_inverse),
        ])

rule normalize_img:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted.btf"
    params:
        function = "stack_normalise",
        alpha = config["alpha"],
        beta = config["beta"]
    output:
        normalised_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            params.function,
            '-i', str(input.input_img),
            '-o', str(output.normalised_img),
            '-a', str(params.alpha),
            '-b', str(params.beta),
        ])

rule worm_unet:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf"
    params:
        function  = "unet_segmentation_stack",
        weights_path = config["worm_unet_weights"],
        #network_name = config["worm_unet_network_name"]# '5358068_1'
    output:
        worm_unet_prediction = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            str(params.function),
            '-i', str(input.input_img),
            '-o', str(output.worm_unet_prediction),
            '-w', str(params.weights_path),
        ])

rule binarize:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented.btf"
    params:
        function = "stack_make_binary",
        threshold = config["threshold"],
        max_value = config["max_value"]
    output:
        binary_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            str(params.function),
            '-i', str(input.input_img),
            '-o', str(output.binary_img),
            '-th', str(params.threshold),
            '-max_val', str(params.max_value),
        ])

rule coil_unet:
    input:
        binary_input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask.btf",
        raw_input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf"
    params:
        function = "unet_segmentation_contours_with_children",
        weights_path= config["coil_unet_weights"]
        #network_name="5910044_0"
    output:
        coil_unet_prediction = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            str(params.function),
            '-bi', str(input.binary_input_img),
            '-ri', str(input.raw_input_img),
            '-o', str(output.coil_unet_prediction),
            '-w', str(params.weights_path),
        ])

rule binarize_coil:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented.btf"
    params:
        function = "stack_make_binary",
        threshold = config["coil_threshold"], # 240
        max_value = config["coil_new_value"] # 255
    output:
        binary_img = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented_mask.btf")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            str(params.function),
            '-i', str(input.input_img),
            '-o', str(output.binary_img),
            '-th', str(params.threshold),
            '-max_val', str(params.max_value),
        ])

rule tiff2avi:
    input:
        input_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.btf"
    params:
        function = "tiff2avi",
        fourcc = config["fourcc"], #"0",
        fps = config["fps"] # "167"
    output:
        avi = _cleanup_helper(f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.avi")
    run:
        from imutils.src import imutils_parser_main

        imutils_parser_main.main([
            params.function,
            '-i', str(input.input_img),
            '-o',str(output.avi),
            '-fourcc', str(params.fourcc),
            '-fps', str(params.fps),
        ])

rule dlc_analyze_videos:
    input:
        input_avi = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised.avi"
    params:
        dlc_model_configfile_path = config["dlc_model_configfile_path"],
        dlc_network_string = config["dlc_network_string"], # Is this used?
        dlc_conda_env = config["dlc_conda_env"]
    output:
        hdf5_file = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised"+config["dlc_network_string"]+".h5"
    shell:
        """
        source /apps/conda/miniconda3/bin/activate {params.dlc_conda_env}
        python -c "from centerline_behavior_annotation.dlc_utils import cluster_analyze_videos; cluster_analyze_videos.main(['-path_config_file', '{params.dlc_model_configfile_path}', '-videofile_path', '{input.input_avi}'])"
        """

rule create_centerline:
    input:
        input_binary_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask_coil_segmented_mask.btf",
        hdf5_file = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised"+config["dlc_network_string"]+".h5"

    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
        number_of_neighbours = "1",
        nose = config['nose'],
        tail = config['tail'],
        num_splines = config['num_splines'],
        fill_with_DLC = "1"
    output :
        output_skel_X = f"{output_behavior_dir}/skeleton_skeleton_X_coords.csv",
        output_skel_Y = f"{output_behavior_dir}/skeleton_skeleton_Y_coords.csv",
        output_spline_K = f"{output_behavior_dir}/skeleton_spline_K.csv",
        output_spline_X = f"{output_behavior_dir}/skeleton_spline_X_coords.csv",
        output_spline_Y = f"{output_behavior_dir}/skeleton_spline_Y_coords.csv",
        corrected_head = f"{output_behavior_dir}/skeleton_corrected_head_coords.csv",
        corrected_tail = f"{output_behavior_dir}/skeleton_corrected_tail_coords.csv"
    run:
        from centerline_behavior_annotation.centerline.dev import head_and_tail

        head_and_tail.main([
            '-i', str(input.input_binary_img),
            '-h5', str(input.hdf5_file),
            '-o', str(params.output_path),
            '-nose', str(params.nose),
            '-tail', str(params.tail),
            '-num_splines', str(params.num_splines),
            '-n', str(params.number_of_neighbours),
            '-dlc', str(params.fill_with_DLC),
        ])

rule invert_curvature_sign:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K.csv",
        # would be good to ad the config yaml file to input because if it is updated the code should re-run
        #config_yaml_file = os.path.join(os.path.dirname(os.path.dirname("{sample}_skeleton_spline_K.csv")), "config.yaml") #hard to get its path
    # params:
    #     output_path = f"{output_behavior_dir}/"
    output:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K_signed.csv"
    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
    run:
        from centerline_behavior_annotation.curvature.src import invert_curvature_sign

        invert_curvature_sign.main([
            '-i', str(params.output_path),
            '-r', str(raw_data_dir),
        ])

rule average_kymogram:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K_signed.csv"
    params:
        #rolling_mean_type =,
        window = config['averaging_window']
    output:
        spline_K_avg = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv"
    run:
        import pandas as pd
        df=pd.read_csv(input.spline_K, index_col=None, header=None)
        df=df.rolling(window=params.window, center=True, min_periods=1).mean()
        df.to_csv(output.spline_K_avg, header=None, index=None)
        print('end of python script')

rule average_xy_coords:
    input:
        spline_X= f"{output_behavior_dir}/skeleton_spline_X_coords.csv",
        spline_Y= f"{output_behavior_dir}/skeleton_spline_Y_coords.csv",
    params:
        #rolling_mean_type =,
        window = config['averaging_window']
    output:
        spline_X_avg= f"{output_behavior_dir}/skeleton_spline_X_coords_avg.csv",
        spline_Y_avg= f"{output_behavior_dir}/skeleton_spline_Y_coords_avg.csv",
    run:
        import pandas as pd

        df=pd.read_csv(input.spline_X, index_col=None, header=None)
        df=df.rolling(window=params.window, center=True, min_periods=1).mean()
        df.to_csv(output.spline_X_avg, header=None, index=None)

        df=pd.read_csv(input.spline_Y, index_col=None, header=None)
        df=df.rolling(window=params.window, center=True, min_periods=1).mean()
        df.to_csv(output.spline_Y_avg, header=None, index=None)

rule hilbert_transform_on_kymogram:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv",
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
        fs = config["sampling_frequency"],
        window = config["hilbert_averaging_window"]
    output:
        # wont be created because they the outputs do not have the {sample} root
        hilbert_regenerated_carrier = f"{output_behavior_dir}/hilbert_regenerated_carrier.csv",
        hilbert_inst_freq = f"{output_behavior_dir}/hilbert_inst_freq.csv",
        hilbert_inst_phase = f"{output_behavior_dir}/hilbert_inst_phase.csv",
        hilbert_inst_amplitude = f"{output_behavior_dir}/hilbert_inst_amplitude.csv"

    #This $DIR only goes one time up
    run:
        from centerline_behavior_annotation.behavior_analysis.src import hilbert_transform

        hilbert_transform.main([
            '-i', str(params.output_path),
            '-kp', str(input.spline_K),
            '-fs', str(params.fs),
            '-w', str(params.window),
        ])

rule fast_fourier_transform:
    input:
        spline_K = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv",
    params:
        # project_folder
        sampling_frequency=config["sampling_frequency"],
        window = config["fft_averaging_window"],
        output_path = f"{output_behavior_dir}/",  # Ulises' functions expect the final slash

    output:
        y_axis_file = f"{output_behavior_dir}/fft_y_axis.csv", #not correct ?
        xf_file = f"{output_behavior_dir}/fft_xf.csv"
    #This $DIR only goes one time up
    run:
        from centerline_behavior_annotation.centerline.dev import fourier_functions

        fourier_functions.main([
            '-i', str(params.output_path),
            '-kp', str(input.spline_K),
            '-fps', str(params.sampling_frequency),
            '-w', str(params.window),
        ])

rule reformat_skeleton_files:
    input:
        spline_K= f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv", #should have signed spline_K
        spline_X= f"{output_behavior_dir}/skeleton_spline_X_coords_avg.csv",
        spline_Y= f"{output_behavior_dir}/skeleton_spline_Y_coords_avg.csv",
        #spline_list = ["{sample}_spline_K.csv", "{sample}_spline_X_coords.csv", "{sample}_spline_Y_coords.csv",]

    output:
        merged_spline_file = f"{output_behavior_dir}/skeleton_merged_spline_data_avg.csv",

    run:
        from centerline_behavior_annotation.centerline.dev import reformat_skeleton_files

        reformat_skeleton_files.main([
            '-i_K', str(input.spline_K),
            '-i_X', str(input.spline_X),
            '-i_Y', str(input.spline_Y),
            '-o', str(output.merged_spline_file),
        ])

rule annotate_behaviour:
    input:
        curvature_file = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv"

    params:
        pca_model_path = config["pca_model"],
        initial_segment = config["initial_segment"],
        final_segment = config["final_segment"],
        window = config["window"]
    output:
        principal_components = f"{output_behavior_dir}/principal_components.csv",
        behaviour_annotation = f"{output_behavior_dir}/beh_annotation.csv"
    run:
        from centerline_behavior_annotation.curvature.src import annotate_reversals_snakemake

        annotate_reversals_snakemake.main([
            '-i', str(input.curvature_file),
            '-pca', str(params.pca_model_path),
            '-i_s', str(params.initial_segment),
            '-f_s', str(params.final_segment),
            '-win', str(params.window),
            '-o_pc', str(output.principal_components),
            '-o_bh', str(output.behaviour_annotation),
        ])

rule annotate_turns:
    input:
        #principal_components = f"{output_behavior_dir}/principal_components.csv"
        spline_K  = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv"
    params:
        output_path = f"{output_behavior_dir}/",  # Ulises' functions expect the final slash
        threshold = config["turn_threshold"],
        initial_segment = config["turn_initial_segment"],
        final_segment = config["turn_final_segment"],
        avg_window = config["turn_avg_window"]
    output:
        turns_annotation = f"{output_behavior_dir}/turns_annotation.csv"

    run:
        from centerline_behavior_annotation.curvature.src import annotate_turns_snakemake

        annotate_turns_snakemake.main([
            '-input', str(input.spline_K),
            '-t', str(params.threshold),
            '-i_s', str(params.initial_segment),
            '-f_s', str(params.final_segment),
            '-avg_window', str(params.avg_window),
            '-bh', str(output.turns_annotation),
        ])

rule self_touch:
    input:
        binary_img = f"{output_behavior_dir}/raw_stack_AVG_background_subtracted_normalised_worm_segmented_mask.btf"
    params:
        external_area = [7000, 20000],
        internal_area = [100, 2000],
    output:
        self_touch = f"{output_behavior_dir}/self_touch.csv"
    run:
        from imutils.src.imfunctions import stack_self_touch
        df = stack_self_touch(input.binary_img, params.external_area, params.internal_area)
        df.to_csv(output.self_touch)

rule calculate_parameters:
    #So far it only calculates speed
    input:
        curvature_file = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv" #This is used as a parameter because it is only used to find the main dir
    output:
        speed_file = f"{output_behavior_dir}/raw_worm_speed.csv" # This is never produced, so this will always run
    params:
        output_path = f"{output_behavior_dir}/", # Ulises' functions expect the final slash
    run:
        from centerline_behavior_annotation.behavior_analysis.src import calculate_parameters

        calculate_parameters.main([
            '-i', str(params.output_path),
            '-r', str(raw_data_dir),
        ])

rule save_signed_speed:
    input:
        raw_speed_file = f"{output_behavior_dir}/raw_worm_speed.csv",
        behaviour_annotation= f"{output_behavior_dir}/beh_annotation.csv"
    output:
        signed_speed_file = f"{output_behavior_dir}/signed_worm_speed.csv" # This is never produced, so this will always run
    run:
        import pandas as pd
        raw_speed_df=pd.read_csv(input.raw_speed_file)
        ethogram_df = pd.read_csv(input.behaviour_annotation)
        signed_speed_df = pd.DataFrame()
        signed_speed_df['Raw Speed Signed (mm/s)'] = raw_speed_df['Raw Speed (mm/s)'] * ethogram_df['0'] * -1 # to invert because fwd is -1 in the ethogram
        signed_speed_df.to_csv(output.signed_speed_file)
        print("If the ethogram had a running average and had less values at the start and end, so will the signed speed")

rule make_behaviour_figure:
    input:
        curvature_file = f"{output_behavior_dir}/skeleton_spline_K_signed_avg.csv",
        pc_file = f"{output_behavior_dir}/principal_components.csv",
        beh_annotation_file = f"{output_behavior_dir}/beh_annotation.csv",
        speed_file = f"{output_behavior_dir}/signed_worm_speed.csv",
        turns_annotation = f"{output_behavior_dir}/turns_annotation.csv"
    output:
        figure = f"{output_behavior_dir}/behavioral_summary_figure.pdf" #This is  never produced, so it whill always run
    params:
        output_path = f"{output_behavior_dir}/",# Ulises' functions expect the final slash
    run:
        from centerline_behavior_annotation.behavior_analysis.src import make_figure

        make_figure.main([
            '-i', str(params.output_path),
            '-r', str(raw_data_dir),
        ])
