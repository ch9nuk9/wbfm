#!/usr/bin/env bash

snakemake --forceall -ns ../pipeline_segmentation.smk --filegraph | dot | display
