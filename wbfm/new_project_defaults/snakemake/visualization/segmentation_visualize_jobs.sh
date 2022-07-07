#!/usr/bin/env bash

snakemake -ns ../pipeline_segmentation.smk --dag | dot | display
