# PWM- and SVM-based models benchmark

# SVM Models' Collection

## Overview
`BENCHMARK.zip` contains everything needed to execute the Snakemake workflow.

## Setup Instructions
Before running the Snakemake pipeline, please ensure the following:

1. Create a Python environment with the required dependencies:
   - pybedtools
   - snakemake
   - tqdm

2. Install lsgkm for training and testing:
   - Clone the [lsgkm](https://github.com/Dongwon-Lee/lsgkm) repository from GitHub.
   - Make sure the folder containing gkmtrain and gkmpredict is added to your `$PATH`.

### General Preparation
Before launching the Snakemake workflow, copy the BED files into the directory `BENCHMARK/ENCODE/ChIP-seq`.

## Command
Once you have prepared the setup you are ready to run the snakemake pipeline:
```bash
snakemake -p --cores <number_of_cores> models_report.csv
