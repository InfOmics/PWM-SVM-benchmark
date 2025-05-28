""" 
"""

import argparse
import subprocess
import os

GKMPREDICT = "gkmpredict"  # lsgkm scoring functionality

def load_svm_model(collection_path: str, model_name: str) -> str:
    if not os.path.isdir(collection_path):
        raise ValueError(f"Collection path {collection_path} is not a directory.")
    model_path = os.path.join(collection_path, f"{model_name}.model.txt")
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file {model_path} does not exist.")
    return model_path

def score_sequences(fasta: str, modelname: str, model: str, outdir: str) -> None:
    if not os.path.isfile(fasta):
        raise ValueError(f"Input FASTA file {fasta} does not exist.")
    if not os.path.isdir(outdir):
        raise ValueError(f"Output directory {outdir} does not exist.")
    # Placeholder for actual scoring logic
    print(f"Scoring sequences in {fasta} using model {model} and saving results to {outdir}")
    scoresfile = os.path.join(outdir, f"{modelname}.scores.txt")
    code = subprocess.call(f"{GKMPREDICT} -T 16 -v 0 {fasta} {model} {scoresfile}", shell=True)
    if code != 0:
        raise subprocess.SubprocessError(f"An error occurred while scoring {fasta}")

def main():
    p = argparse.ArgumentParser(description="Score input sequences in FASTA file using SVM models")
    p.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    p.add_argument("--collection-path", dest="collection_path", type=str, required=True, help="Path to SVM model collection")
    p.add_argument("--model", type=str, required=True, help="Name of the SVM model to use")
    p.add_argument("--output", type=str, required=True, help="Path to output file")
    args = p.parse_args()
    model = load_svm_model(args.collection_path, args.model)
    score_sequences(args.fasta, args.model, model, args.output)


if __name__ == "__main__":
    main()
