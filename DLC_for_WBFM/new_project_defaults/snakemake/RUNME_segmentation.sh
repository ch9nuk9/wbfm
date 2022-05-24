#!/usr/bin/env bash

snakemake -s pipeline_segmentation.smk --cores --latency-wait 600
