#!/usr/bin/env bash

snakemake --forceall -ns post_segmentation_pipeline.smk --filegraph | dot | display
