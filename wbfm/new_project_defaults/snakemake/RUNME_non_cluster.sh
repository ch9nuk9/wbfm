#!/bin/bash

snakemake traces_and_behavior -s pipeline.smk --latency-wait 60 --cores 56
