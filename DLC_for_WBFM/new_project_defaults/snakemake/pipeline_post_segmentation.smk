configfile: "config.yaml"

rule all:
    input:
        expand("{dir}/{test}", test=config['output_4'], dir=config['project_dir'])

#
# Tracklets
#
rule build_frame_objects:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/2a-build_frame_objects.py", code=config['code_path']),
        masks=ancient(rules.segmentation.output),
        files=expand("{dir}/{input}", input=config['input_2a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_2a'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"


rule match_frame_pairs:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/2b-match_adjacent_volumes.py", code=config['code_path']),
        masks=ancient(rules.segmentation.output),
        files=expand("{dir}/{input}", input=config['input_2b'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_2b'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"


rule postprocess_matches_to_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/2c-postprocess_matches_to_tracklets.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_2c'], dir=config['project_dir']),
    output:
        expand("{dir}/{output}", output=config['output_2c'], dir=config['project_dir'])
    threads: 8
    shell:
        "python {input.code_path} with project_path={input.cfg}"

#
# Tracking
#
rule tracking:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/3a-track_using_superglue.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_3a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3a'], dir=config['project_dir'])
    threads: 48
    shell:
        "python {input.code_path} with project_path={input.cfg}"

rule combine_tracking_and_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/3b-match_tracklets_and_tracks_using_neuron_initialization.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_3b'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3b'], dir=config['project_dir'])
    threads: 8
    shell:
        "python {input.code_path} with project_path={input.cfg}"

#
# Traces
#
rule extract_full_traces:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        code_path=expand("{code}/4-make_final_traces.py", code=config['code_path']),
        files=expand("{dir}/{input}", input=config['input_4'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_4'], dir=config['project_dir'])
    threads: 56
    shell:
        "python {input.code_path} with project_path={input.cfg}"
