#!/usr/bin/env bash

snakemake -s post_segmentation_pipeline.smk --cores --latency-wait 600
