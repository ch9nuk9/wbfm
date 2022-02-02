#!/usr/bin/env bash

snakemake --forceall -ns full_pipeline.smk --filegraph | dot | display
