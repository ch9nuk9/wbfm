#!/usr/bin/env bash

snakemake traces --debug-dag -n -s pipeline.smk --cores
