configfile: "config.yaml"

rule all:
    input:
        expand("{test}", test=config['output_4c'])

#
# Preprocessing
#

rule preprocessing:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path'])
    output:
        expand("{dir}/{output}", output=config['output_0'], dir=config['project_dir']),
    shell:
        "python {input.code_path}/alternate/0+build_bounding_boxes.py with project_path={input.cfg}"

#
# Segmentation
#
rule segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        # files=expand("{input}", input=config['input_1'])
    output:
        metadata=expand("{dir}/{output}", output=config['output_1'], dir=config['project_dir']),
        masks=directory(expand("{dir}/{output}", output=config['output_1_dir'], dir=config['project_dir']))
    shell:
        "python {input.code_path}/1-segment_video.py with project_path={input.cfg}"


#
# Tracklets
#
rule match_frame_pairs:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        masks=ancient(rules.segmentation.output)
    output:
        expand("{dir}/{output}", output=config['output_2a'], dir=config['project_dir'])
    shell:
        "python {input.code_path}/2a-pairwise_match_sequential_frames.py with project_path={input.cfg}"


rule postprocess_matches_to_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_2b'], dir=config['project_dir']),
    output:
        expand("{dir}/{output}", output=config['output_2b'], dir=config['project_dir'])
    shell:
        "python {input.code_path}/2b-postprocess_matches_to_tracklets.py with project_path={input.cfg}"


rule reindex_segmentation_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_2c'], dir=config['project_dir']),
        masks=ancient(rules.segmentation.output.masks)
    output:
        directory(expand("{dir}/{output}", output=config['output_2c_dir'], dir=config['project_dir']))
    shell:
        "python {input.code_path}/2c-reindex_segmentation_training_masks.py with project_path={input.cfg}"


rule save_training_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_2d'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_2d'], dir=config['project_dir'])
    shell:
        "python {input.code_path}/2d-save_training_tracklets_as_dlc.py with project_path={input.cfg}"


#
# Tracking
#

rule fndc_tracking:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_3a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3a'], dir=config['project_dir'])
    shell:
        "python {input.code_path}/alternate/3-track_using_fdnc.py with project_path={input.cfg}"

rule combine_tracking_and_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_3b'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3b'], dir=config['project_dir'])
    shell:
        "python {input.code_path}/postprocessing/3c+combine_tracklets_and_dlc_tracks.py with project_path={input.cfg}"

#
# Traces
#

rule match_tracks_and_segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_4a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_4a'], dir=config['project_dir'])
    shell:
        "python {input.code_path}/4a-match_tracks_and_segmentation.py with project_path={input.cfg}"

rule reindex_segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_4b'], dir=config['project_dir']),
        masks=ancient(rules.segmentation.output.masks)
    output:
        directory(expand("{dir}/{output}", output=config['output_4b_dir'], dir=config['project_dir']))
    shell:
        "python {input.code_path}/4b-reindex_segmentation_full.py with project_path={input.cfg}"

rule extract_full_traces:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_4c_dir'], dir=config['project_dir'])
    output:
        rules.reindex_segmentation.output
    shell:
        "python {input.code_path}/4c-extract_full_traces.py with project_path={input.cfg}"
