#!/usr/bin/env bash

snakemake --debug-dag -n -s pipeline_post_segmentation.smk --cores
