#!/usr/bin/env bash

snakemake -ns ../pipeline_post_segmentation.smk --dag | dot | display
