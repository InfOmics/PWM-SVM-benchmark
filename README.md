# PWM- and SVM-based models benchmark

## Overview
The `benchmark/src` folder contains scripts for replicating the analysis:  
- `dataset_split.py` – Generates train and test datasets for each input BED file.  
- `compute_benchmark_datasets.py` – Refines datasets for selected comparisons.  
- `models_training.py` – Trains PWM- and SVM-based models on refined datasets.  
- `score_models.py` – Computes performance scores and tables for trained models.  

Use `pipeline.py` to run the entire analysis, selecting comparisons (`width`, `size`, `optimal-local`, `optimal-global`) to replicate.

## Setup Instructions
1. Create a Conda environment with Python, then install dependencies with:  
   ```bash
   pip install -r requirements.txt
   ```
   
2. Install [MEME-SUITE](https://meme-suite.org/meme/doc/install.html?man_type=web) and [lsgkm](https://github.com/Dongwon-Lee/lsgkm)
   - Make sure `~/meme/bin` and `~/lsgkm/bin` are correctly added to your `$PATH`.

3. Modify `SRC_DIR`, `REFERENCE_GENOME`, `POSITIVE_BEDS`, `NEGATIVE_BEDS` inside `pipeline.py` to reflect your files organization.

## Usage
Run the pipeline with:

```bash
python pipeline.py [<comparisons>]
```
`<comparisons>` can be any combination of `width`, `size`, `optimal-local`, `optimal-global`, or `all` to run everything.

All the results will be included in a `benchmark` folder.

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

## Usage
Once you have prepared the setup you are ready to run the snakemake pipeline:
```bash
snakemake -p --cores <number_of_cores> models_report.csv
