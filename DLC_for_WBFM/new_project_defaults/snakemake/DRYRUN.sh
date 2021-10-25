#!/usr/bin/env bash

snakemake --debug-dag -n -s full_pipeline.smk --cores
