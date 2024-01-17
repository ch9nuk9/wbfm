configfile: "config.yaml"

def _run_helper(script_name, project_path):
    import importlib
    _module = importlib.import_module(f"wbfm.scripts.{script_name}")
    _module.ex.run(config_updates=dict(project_path=project_path))


rule all:
    input:
        expand("{dir}/{test}", test=config['output_4'], dir=config['project_dir'])

rule preprocessing:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
    output:
        expand("{dir}/{output}", output=config['output_0'], dir=config['project_dir']),
    run:
        _run_helper(config['script_0'], str(input.cfg))

#
# Segmentation
#
rule segmentation:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        files=expand("{dir}/{input}", input=config['input_1'], dir=config['project_dir'])
    output:
        metadata=expand("{dir}/{output}", output=config['output_1'], dir=config['project_dir']),
        masks=directory(expand("{dir}/{output}", output=config['output_1_dir'], dir=config['project_dir']))
    threads: 56
    run:
        _run_helper(config['script_1'], str(input.cfg))


#
# Tracklets
#
rule build_frame_objects:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        masks=ancient(expand("{dir}/{input}", input=config['output_1_dir'], dir=config['project_dir'])),
        files=expand("{dir}/{input}", input=config['input_2a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_2a'], dir=config['project_dir'])
    threads: 56
    run:
        _run_helper(config['script_2a'], str(input.cfg))


rule match_frame_pairs:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        masks=ancient(expand("{dir}/{input}", input=config['output_1_dir'], dir=config['project_dir'])),
        files=expand("{dir}/{input}", input=config['input_2b'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_2b'], dir=config['project_dir'])
    threads: 56
    run:
        _run_helper(config['script_2b'], str(input.cfg))


rule postprocess_matches_to_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        files=expand("{dir}/{input}", input=config['input_2c'], dir=config['project_dir']),
    output:
        expand("{dir}/{output}", output=config['output_2c'], dir=config['project_dir'])
    threads: 8
    run:
        _run_helper(config['script_2c'], str(input.cfg))

#
# Tracking
#
rule tracking:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        files=expand("{dir}/{input}", input=config['input_3a'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3a'], dir=config['project_dir'])
    threads: 48
    run:
        _run_helper(config['script_3a'], str(input.cfg))

rule combine_tracking_and_tracklets:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        files=expand("{dir}/{input}", input=config['input_3b'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_3b'], dir=config['project_dir'])
    threads: 8
    run:
        _run_helper(config['script_3b'], str(input.cfg))

#
# Traces
#
rule extract_full_traces:
    input:
        cfg=expand("{dir}/project_config.yaml", dir=config['project_dir']),
        files=expand("{dir}/{input}", input=config['input_4'], dir=config['project_dir'])
    output:
        expand("{dir}/{output}", output=config['output_4'], dir=config['project_dir'])
    threads: 56
    run:
        _run_helper(config['script_4'], str(input.cfg))
