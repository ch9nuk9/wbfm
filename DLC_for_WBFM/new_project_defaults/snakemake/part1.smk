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
    shell:
        "python {input.code_path}/1-segment_video.py with project_path={input.cfg}"


rule training_data:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_2'])
    output:
        expand("{output}", output=config['output_2'])
    shell:
        "python {input.code_path}/2-produce_training_data.py with project_path={input.cfg}"


