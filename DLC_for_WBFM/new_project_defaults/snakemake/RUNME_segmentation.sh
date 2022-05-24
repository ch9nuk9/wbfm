#!/usr/bin/env bash

snakemake -s segmentation_pipeline.smk --cores --latency-wait 600
