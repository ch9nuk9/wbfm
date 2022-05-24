#!/usr/bin/env bash

snakemake -ns segmentation_pipeline.smk --dag | dot | display
