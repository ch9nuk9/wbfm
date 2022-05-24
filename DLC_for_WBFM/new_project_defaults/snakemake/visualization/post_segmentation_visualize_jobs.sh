#!/usr/bin/env bash

snakemake -ns post_segmentation_pipeline.smk --dag | dot | display
