#!/usr/bin/env bash

snakemake -s full_pipeline.smk save_training_tracklets --cores
