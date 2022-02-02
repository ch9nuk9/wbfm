configfile: "config.yaml"

rule all:
    input:
        expand("{dir}/{test}", test=config['output_4c'], dir=config['project_dir'])

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
        code_path=expand("{code}/1-segment_video.py", code=config['code_path']),
        # files=expand("{input}", input=config['input_1'])
    output:
        metadata=expand("{dir}/{output}", output=config['output_1'], dir=config['project_dir']),
        masks=directory(expand("{dir}/{output}", output=config['output_1_dir'], dir=config['project_dir']))
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"


#
# Tracklets
#
rule match_frame_pairs:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/2a-pairwise_match_sequential_frames.py", code=config['code_path']),
        masks=ancient(rules.segmentation.output)
    output:
        expand("{dir}/{output}", output=config['output_2a'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"


rule postprocess_matches_to_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/2b-postprocess_matches_to_tracklets.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_2b'], dir=config['project_dir']),
    output:
        expand("{dir}/{output}", output=config['output_2b'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"


rule reindex_segmentation_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/2c-reindex_segmentation_training_masks.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_2c'], dir=config['project_dir']),
        masks=ancient(rules.segmentation.output.masks)
    output:
        directory(expand("{dir}/{output}", output=config['output_2c_dir'], dir=config['project_dir']))
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"

#
# Tracking
#
rule fndc_tracking:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/3a-track_using_fdnc.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_3a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3a'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"

rule combine_tracking_and_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/3b-match_tracklets_and_tracks_using_neuron_initialization.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_3b'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3b'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"

#
# Traces
#
rule match_tracks_and_segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/4a-match_tracks_and_segmentation.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_4a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_4a'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"

rule reindex_segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/4b-reindex_segmentation_full.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_4b'], dir=config['project_dir']),
        masks=ancient(rules.segmentation.output.masks)
    output:
        directory(expand("{dir}/{output}", output=config['output_4b_dir'], dir=config['project_dir']))
    threads: 8
    shell:
        "python {input.code_path} with project_path={input.cfg}"

rule extract_full_traces:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/4c-extract_full_traces.py", code=config['code_path']),
        masks=ancient(rules.reindex_segmentation.output)
    output:
        expand("{dir}/{output}", output=config['output_4c'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"
