#!/usr/bin/env bash

snakemake --debug-dag -n -s pipeline_segmentation.smk --cores
