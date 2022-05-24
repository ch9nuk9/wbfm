#!/usr/bin/env bash

snakemake --forceall -ns segmentation_pipeline.smk --filegraph | dot | display
