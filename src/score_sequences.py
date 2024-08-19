"""
"""

from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

import subprocess
import sys
import os

MODELTYPES = ["svm", "pwm"]
GKMPREDICT = "gkmpredict"
FIMO = "fimo"
FIMODEF = "--max-stored-scores 1000000000 --verbosity 1 --thresh 1"
COLNAMES = ["SEQNAME", "SCORE", "PEAK"]

def parse_commandline(args: List[str]) -> Tuple[str, str, str, str, str]:
    if len(args) != 5:
        raise ValueError("Too many/few input arguments")
    positive, negative, model, modeltype, outdir = args
    if not os.path.isfile(positive):
        raise FileNotFoundError(f"Cannot find {positive}")
    if not os.path.isfile(negative):
        raise FileNotFoundError(f"Cannot find {negative}")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Cannot find {model}")
    if modeltype not in MODELTYPES:
        raise ValueError(f"Forbidden model type ({modeltype})")
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f"Cannot find {outdir}")
    return positive, negative, model, modeltype, outdir

def gkmpredict(seqfile: str, model: str, scoresfile: str) -> str:
    try:
        code = subprocess.call(
            f"{GKMPREDICT} -T 16 {seqfile} {model} {scoresfile}", 
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.DEVNULL,
        )
        assert code == 0 and os.stat(scoresfile).st_size > 0
    except OSError as e:
        raise OSError(f"An error occurred while scoring {seqfile} (svm)") from e
    return scoresfile

def store_scores_svm(scorespos_fname: str, scoresneg_fname: str, scores_prefix: str) -> None:
    scorespos = pd.read_csv(scorespos_fname, sep="\t", header=None)
    scoresneg = pd.read_csv(scoresneg_fname, sep="\t", header=None)
    # rename columns
    scorespos.columns = COLNAMES[:2]
    scorespos[COLNAMES[2]] = 1
    scoresneg.columns = COLNAMES[:2]
    scoresneg[COLNAMES[2]] = 0
    scores = pd.concat([scorespos, scoresneg])  # concatenate reports
    # shuffle and store report
    scores = scores.sample(frac=1).to_csv(
        f"{scores_prefix}_svm.tsv", sep="\t", index=False
    )
    # remove tmp files
    code = subprocess.call(f"rm {scoresneg_fname} {scorespos_fname}", shell=True)
    if code != 0:
        raise subprocess.SubprocessError(
            "An error occurred while removing tmp score files"
        )

def score_svm(testpos: str, testneg: str, model: str, outdir: str, scores_prefix: str) -> None:
    scorespos = os.path.join(outdir, f"{scores_prefix}.pos.scores")  # positive seqs
    scoresneg = os.path.join(outdir, f"{scores_prefix}.neg.scores")  # negative seqs
    # score positive and negative sequences 
    store_scores_svm(
        gkmpredict(testpos, model, scorespos), 
        gkmpredict(testneg, model, scoresneg),
        scores_prefix
    )

def read_fasta(fastafile: str) -> Dict[str, str]:
    try:
        with open(fastafile, mode="r") as infile:
            seqnames = [line.strip()[1:] for line in infile if line.startswith(">")]
        with open(fastafile, mode="r") as infile:
            sequences = [line.strip() for line in infile if not line.startswith(">")]
    except IOError as e:
        raise OSError(f"Parsing {fastafile} failed") from e
    return {seqname: sequences[i] for i, seqname in enumerate(seqnames)}


def fimo(seqfile: str, working_dir: str, model: str) -> Dict[str, float]:
    fasta = read_fasta(seqfile)  # read sequences
    scores = {seqname: None for seqname in fasta}
    for seqname in fasta:
        fname = os.path.join(working_dir, seqname.replace(":", "_").replace("-", "_"))
        try:
            with open(f"{fname}.fa", mode="w") as outfile:
                outfile.write(f">{seqname}\n{fasta[seqname]}\n")
            code = subprocess.call(
                f"{FIMO} --oc {fname} {FIMODEF} {model} {fname}.fa",
                shell=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.DEVNULL,
            )
            assert code == 0
        except OSError as e:
            raise OSError(f"An error occurred while scoring {seqfile} (pwm)") from e
        x = pd.read_csv(os.path.join(fname, "fimo.tsv"), sep="\t")
        code = subprocess.call(f"rm -rf {fname} {fname}.fa", shell=True)
        if code != 0:
            raise subprocess.SubprocessError(
                "An error occurred while removing tmp score files"
            )
        scores[seqname] = max(x["score"].tolist())
    return scores


def score_pwm(testpos: str, testneg: str, model: str, outdir: str, scores_prefix: str) -> None:
    working_dir = os.path.dirname(outdir)
    # score positive and negative sequences
    scorespos = fimo(testpos, working_dir, model)
    scoresneg = fimo(testneg, working_dir, model)



def main():
    testpos, testneg, model, modeltype, outdir = parse_commandline(sys.argv[1:])
    scores_prefix = os.path.basename(testpos).split("_")[0]

