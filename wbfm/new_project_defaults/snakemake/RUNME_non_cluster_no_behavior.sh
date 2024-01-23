#!/bin/bash

snakemake traces -s pipeline.smk --latency-wait 60 --cores 56
