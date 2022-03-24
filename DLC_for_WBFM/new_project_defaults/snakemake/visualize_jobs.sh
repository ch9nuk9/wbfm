#!/usr/bin/env bash

snakemake -ns full_pipeline.smk --dag | dot | display
