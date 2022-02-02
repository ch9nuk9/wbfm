#!/usr/bin/env bash

snakemake -s full_pipeline.smk --cores --latency-wait 600
