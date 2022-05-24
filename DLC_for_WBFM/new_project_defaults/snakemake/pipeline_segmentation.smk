configfile: "config.yaml"

rule all:
    input:
        expand("{dir}/{test}", test=config['output_1'], dir=config['project_dir'])

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
        "python {input.code_path}/0b-preprocess_working_copy_of_data.py with project_path={input.cfg}"

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
