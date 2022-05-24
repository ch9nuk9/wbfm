#!/usr/bin/env bash

snakemake -s pipeline_post_segmentation.smk --cores --latency-wait 600
