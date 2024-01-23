#!/usr/bin/env bash

snakemake traces_and_behavior --debug-dag -n -s pipeline.smk --cores
