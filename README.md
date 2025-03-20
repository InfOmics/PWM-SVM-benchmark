# PWM and SVM-based Transcription Factor Motif Models Benchmark

## Overview
The `src/benchmark` folder contains scripts for reproducing the benchmark analyses presented in the paper *"Benchmarking PWM and SVM-based Models for Transcription Factor Binding Site Prediction: A Comparative Analysis on Synthetic and Biological Data."* These scripts automate dataset preparation, model training, and performance evaluation for Position Weight Matrix (PWM) and Support Vector Machine (SVM) approaches.

### Available Scripts

- `dataset_split.py` – Generates training and testing datasets from each input BED file, ensuring proper data partitioning for benchmarking.

- `compute_benchmark_datasets.py` – Processes and refines the datasets, filtering and structuring them to enable model comparisons.

- `models_training.py` – Trains PWM and SVM-based motif models on the refined datasets, preparing them for performance assessment.

- `score_models.py` – Computes performance metrics (AUPRC, AUROC, F1) and generates tables summarizing the predictive accuracy of the trained models.

### Running the Benchmark Pipeline

To execute the entire benchmarking workflow, use `pipeline.py`, which automates dataset preparation, model training, and evaluation. You can specify different benchmarking comparisons, including:

- `width` – Evaluates model performance across varying sequence window sizes.

- `size` – Assesses the impact of training dataset size on models' predictive accuracy.

- `optimal-local` – Compares models using locally optimized (experiment-specific) hyperparameters.

- `optimal-global` – Compares models with globally optimized (tool-specific) parameters for broader generalization.

This pipeline ensures reproducibility of the reported findings and allows further customization for additional experiments.

## Setup Instructions

To set up the environment and dependencies for running the benchmark pipeline, follow these steps:

#### 1. Create a `Conda/Mamba` environment 

It is recommended to use `Conda` or `Mamba` to manage dependencies efficiently. Create a new environment with Python:
```shell
conda create -n <env_name> python
```
or, if using `Mamba` for faster package resolution:
```shell
mamba create -n <env_name> python
```

Activate the environment:
```shell
conda activate <env_name>
```

#### 2. Clone the Code Repository and Install Required Python Packages

Download the project source code by cloning the Git repository:
```shell
git clone https://github.com/InfOmics/PWM-SVM-benchmark.git
cd PWM-SVM-benchmark
```

After activating the environment, install the required dependencies:
```shell
pip install -r requirements.txt
```

#### 3. Install External Dependencies

**MEME-SUITE**

[MEME-SUITE](https://meme-suite.org/meme/doc/install.html?man_type=web) is required for PWM-based motif analysis. Install it by following these steps:
```shell
conda install -c bioconda meme
```
or, if using `mamba`:
```shell
mamba install -c bioconda meme
```

**LS-GKM**

[LS-GKM](https://github.com/Dongwon-Lee/lsgkm) is needed for SVM-based models training. Install it with:
```shell
conda install -c bioconda ls-gkm
```
or, if using `mamba`:
```shell
mamba install -c bioconda ls-gkm
```
#### 4. Configure Pipeline Paths

Before running the benchmark pipeline, update `pipeline.py` to match your dataset and file structure. Modify the following variables inside `pipeline.py`:

- `SRC_DIR` – The directory containing source scripts and models.

- `REFERENCE_GENOME` – Path to the reference genome FASTA file.

- `POSITIVE_BEDS` – List of BED files with known transcription factor binding sites (TFBS).

- `NEGATIVE_BEDS` – List of BED files containing control/background regions.

Example modification in `pipeline.py`:
```python
SRC_DIR = "/path/to/src"
REFERENCE_GENOME = "/path/to/reference/genome.fasta"
POSITIVE_BEDS = "/path/to/positive_data_bed/"
NEGATIVE_BEDS = "/path/to/negative_data_bed"
```

Once these steps are completed, you can proceed with running the benchmark pipeline.

## Usage

The benchmarking pipeline automates dataset preparation, model training, and evaluation. Follow the instructions below to execute the pipeline and analyze the results.

### Running the Benchmark Pipeline

To launch the full benchmarking process, use the following command:
```bash
python pipeline.py [<comparisons>]
```

where `<comparisons>` specifies which benchmarking experiments to run. You can choose one or more of the following:

- `width` – Evaluates model performance across different sequence window sizes.

- `size` – Assesses the impact of training dataset size on predictive accuracy.

- `optimal-local` – Compares models using locally optimized (experiment-specific) hyperparameters.

- `optimal-global` – Compares models trained with globally optimized (tool-specific) parameters for broader generalization.

- `all` – Runs all available comparisons sequentially.

### Example Commands

Run all benchmark comparisons:
```bash
python pipeline.py all
```

Run only the size and optimal-local comparisons:
```bash
python pipeline.py size optimal-local
```

### Understanding the Output
After execution, all results will be stored in the `benchmark/` directory. The pipeline generates the following outputs:

- `benchmark/scores/` – Contains sequence scores, computed from using the trained models 

- `benchmark/performance` – Contains accuracy metrics, and summary tables.

- `benchmark/models/` – Saves trained PWM and SVM models for reproducibility.

# SVM-based Motif Models Collection

## Overview

The `src/svm-models-database` directory contains all necessary resources to execute the Snakemake workflow for training and benchmarking Support Vector Machine (SVM)-based motif models. This workflow automates the process of:

- **Data Preparation** – Organizing ChIP-seq datasets and processing BED files.

- **Model Training** – Using gkm-SVM (gapped k-mer SVM) to train motif prediction models.

- **Model Evaluation** – Benchmarking trained models based on various performance metrics.

- **Result Reporting** – Generating structured outputs, including trained models, performance reports, and logs.

## Setup Instructions

Before running the Snakemake pipeline, follow these steps to ensure a proper setup.

#### Step 1: Set Up a Python Environment

It is recommended to create a dedicated Python environment to manage dependencies.

Using Conda (Recommended)
```bash
conda create -n svm_motif_env python=3.9
conda activate svm_motif_env
pip install pybedtools snakemake tqdm
```
or, if using Mamba (Recommended)
```bash
mamba create -n svm_motif_env python=3.9
mamba activate svm_motif_env
pip install pybedtools snakemake tqdm
```

#### Step 2: Install LS-GKM for Model Training and Testing

The LS-GKM software is required for training SVM-based motif models. Install it with:
```shell
conda install -c bioconda ls-gkm
```
or, if using `mamba`:
```shell
mamba install -c bioconda ls-gkm
```

#### Step 3: Validate the Setup
Ensure all required dependencies are correctly installed by running:
```bash
python -c "import pybedtools, tqdm; print('All dependencies installed successfully!')"
snakemake --help
gkmtrain -h
gkmpredict -h
```

Once these steps are completed, your environment is ready to execute the Snakemake workflow. 

## Usage

Once the setup is complete, you can execute the Snakemake pipeline to train and benchmark SVM-based motif models.

### Basic Usage
To run the pipeline with a specified number of CPU cores, use:
```bash
snakemake -p --cores <number_of_cores> models_report.csv
```

Replace `<number_of_cores>` with the number of CPU cores available on your machine. More cores allow faster execution.

### Pipeline Execution Steps

1. **Prepares Input Data**

- Reads ChIP-seq BED files and processes them for training.

- Extracts sequence features and prepares positive/negative datasets.

2. **Trains SVM-based Motif Models**

- Uses `gkmtrain` to build predictive models.

3. **Evaluates Model Performance**

- Predicts binding sites using gkmpredict.

- Computes AUC-ROC, precision, recall, and F1-score.

4. **Generates Final Reports**

- Saves trained models in the results/models/ directory.

- Creates a CSV summary file (`models_report.csv`) with performance metrics.

### Example: Running with 8 Cores
```bash
Copy code
snakemake -p --cores 8 models_report.csv
```

## Citation

Stay tuned for more 

## License

MIT