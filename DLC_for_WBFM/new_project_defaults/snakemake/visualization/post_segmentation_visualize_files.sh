#!/usr/bin/env bash

snakemake --forceall -ns ../pipeline_post_segmentation.smk --filegraph | dot | display
