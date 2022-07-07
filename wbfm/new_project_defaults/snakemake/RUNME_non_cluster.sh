#!/bin/bash

snakemake -s pipeline.smk --latency-wait 60 --cores 56
