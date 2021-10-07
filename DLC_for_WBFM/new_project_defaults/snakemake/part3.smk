configfile: "config.yaml"

rule all:
    input:
        expand("{test}", test=config['output_4'])

# Note: the files that I need to be there will not be passed directly, but will be read
rule step4:
    input:
        cfg=expand("{cfg}", cfg=config['project_path']),
        code_path=expand("{code}", code=config['code_path']),
        files=expand("{input}", input=config['input_4'])
    output:
        expand("{output}", output=config['output_4'])
    shell:
        "python {input.code_path}/4-make_full_traces with project_path={input.cfg}"


