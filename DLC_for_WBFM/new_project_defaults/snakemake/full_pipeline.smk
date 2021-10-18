configfile: "config.yaml"

rule all:
    input:
        expand("{test}", test=config['output_2'])

###
### Segmentation
###
rule segmentation:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_1'])
    output:
        expand("{output}", output=config['output_1'])
        directory(expand("{output}", output=config['output_1_dir']))
    shell:
        "python {input.code_path}/1-segment_video.py with project_path={input.cfg}"


###
### Tracklets
###
rule training_data_a:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_2a'])
    output:
        expand("{output}", output=config['output_2a'])
    shell:
        "python {input.code_path}/2a-make_short_tracklets.py with project_path={input.cfg}"


rule training_data_b:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_2b'])
    output:
        directory(expand("{output}", output=config['output_2b_dir']))
    shell:
        "python {input.code_path}/2b-reindex_segmentation_training.py with project_path={input.cfg}"


rule training_data_c:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_2c'])
    output:
        expand("{output}", output=config['output_2c'])
    shell:
        "python {input.code_path}/2c-save_training_tracklets_as_dlc.py with project_path={input.cfg}"


###
### Tracking
###

rule tracking_a:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_3a'])
    output:
        expand("{output}", output=config['output_3a'])
    shell:
        "python {input.code_path}/alternate/3-track_using_fdnc.py with project_path={input.cfg}"

rule tracking_b:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_3b'])
    output:
        expand("{output}", output=config['output_3b'])
    shell:
        "python {input.code_path}/postprocessing/3c+combine_tracklets_and_dlc_tracks.py with project_path={input.cfg}"

###
### Traces
###

rule traces_a:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_4a'])
    output:
        expand("{output}", output=config['output_4a'])
    shell:
        "python {input.code_path}/4a-match_tracks_and_segmentation.py with project_path={input.cfg}"

rule traces_b:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_4b'])
    output:
        directory(expand("{output}", output=config['output_4b_dir']))
    shell:
        "python {input.code_path}/4b-reindex_segmentation_full.py with project_path={input.cfg}"

rule traces_c:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_4c'])
    output:
        expand("{output}", output=config['output_4c'])
    shell:
        "python {input.code_path}/4c-extract_full_traces.py with project_path={input.cfg}"
