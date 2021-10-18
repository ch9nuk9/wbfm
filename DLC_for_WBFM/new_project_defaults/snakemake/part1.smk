configfile: "config.yaml"

rule all:
    input:
        expand("{test}", test=config['output_2'])

# Note: the files that I need to be there will not be passed directly, but will be read
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


rule training_data_a:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_2a'])
    output:
        expand("{output}", output=config['output_2a'])
        directory(expand("{output}", output=config['output_2_dir']))
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
        directory(expand("{output}", output=config['output_2c']))
    shell:
        "python {input.code_path}/2c-save_training_tracklets_as_dlc.py with project_path={input.cfg}"
