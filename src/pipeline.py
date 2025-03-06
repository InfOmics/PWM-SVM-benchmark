import sys
import subprocess
from typing import List, Set
from tqdm import tqdm
from time import time
import os

# Configuration constants
SRC_DIR = "path/to/src_dir"
REFERENCE_GENOME = "path/to/ref_genome"
POSITIVE_BEDS = "path/to/positive_bedfiles"
NEGATIVE_BEDS = "path/to/negative_bedfiles"
BENCHMARK_DIR = "benchmark"
DATADIR = f"{BENCHMARK_DIR}/data"
TRAINDATADIR = f"{DATADIR}/traindata"
DATASETDIR = f"{BENCHMARK_DIR}/comparison-data"

# Valid comparisons
VALID_COMPARISONS = ["size", "width", "optimal-global", "optimal-local"]
BASIC_COMPARISONS = {"size", "width"}  # Comparisons that must be run before advanced ones

def run_command(cmd: str, task_name: str) -> None:
    start_time = time()
    exit_code = subprocess.call(cmd, shell=True)
    assert exit_code == 0, f"Error during {task_name}: {cmd}"
    elapsed_time = time() - start_time
    print(f"{task_name} completed in {elapsed_time:.2f} seconds.")

def compute_datasets(comparison: str) -> None:
    datasetdir = DATASETDIR if comparison in BASIC_COMPARISONS else os.path.dirname(DATASETDIR)
    cmd = f"python {SRC_DIR}/compute_benchmark_datasets.py {comparison} {TRAINDATADIR} {REFERENCE_GENOME} {datasetdir}"
    run_command(cmd, f"Benchmark dataset creation")

def train_models(comparison: str) -> None:
    cmd = f"python {SRC_DIR}/models_training.py {comparison} {DATASETDIR} {BENCHMARK_DIR}"
    run_command(cmd, f"Models training")

def test_models(comparison: str) -> None:
    cmd = f"python {SRC_DIR}/score_models.py {comparison} {DATADIR} {BENCHMARK_DIR}"
    run_command(cmd, f"Models scoring")

def parse_commandline(args: List[str]) -> Set[str]:
    if not args:
        raise ValueError("No comparisons specified. Use 'all' or a combination of: size, width, optimal-global, optimal-local.")

    if args[0].lower() == "all":
        return VALID_COMPARISONS  # Run all comparisons

    comparisons = set(args)
    invalid_comparisons = comparisons - set(VALID_COMPARISONS)
    if invalid_comparisons:
        raise ValueError(f"Invalid comparisons: {invalid_comparisons}. Valid options are: {VALID_COMPARISONS}.")

    # Ensure that optimal-global and optimal-local are only run if size and width are included
    advanced_comparisons = {"optimal-global", "optimal-local"}
    if (advanced_comparisons & comparisons) and not (BASIC_COMPARISONS <= comparisons):
        raise ValueError("'optimal-global' and 'optimal-local' can only be run if 'size' and 'width' are also specified.")
    
    # Reorder comparisons to ensure 'size' and 'width' come first
    ordered_comparisons = list(BASIC_COMPARISONS & comparisons) + list(comparisons - BASIC_COMPARISONS)
    return ordered_comparisons

def main():

    print("Starting analysis pipeline...")

    try:
        comparisons = parse_commandline(sys.argv[1:])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 1: Split dataset
    print("Splitting dataset...")
    cmd = f"python {SRC_DIR}/dataset_split.py {POSITIVE_BEDS} {NEGATIVE_BEDS} {REFERENCE_GENOME} {DATADIR}"
    run_command(cmd, "Dataset splitting")

    # Run
    for comparison in comparisons:
        print(f"Executing {comparison}")

        print("Computing benchmark datasets.")
        compute_datasets(comparison)

        print("Training models...")
        train_models(comparison)

        print("Testing models...")
        test_models(comparison)

    print("Analysis pipeline completed successfully.")

if __name__ == "__main__":
    main()