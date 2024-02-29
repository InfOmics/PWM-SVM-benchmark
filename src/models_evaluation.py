"""
"""

from typing import Tuple

import pandas as pd
import numpy as np

import subprocess
import sys
import os


def read_scores(scorespos: str, scoresneg: str, modeltype: str) -> pd.DataFrame:
    if modeltype not in ["pwm", "svm"]:
        raise ValueError("Unknown model type")
    header = None if modeltype == "svm" else 0
    scorespos = pd.read_csv(scorespos, header=header, sep="\t")
    scoresneg = pd.read_csv(scoresneg, header=header, sep="\t")
    if modeltype == "pwm":  # erase colnames
        scorespos.columns = list(range(scorespos.shape[1]))
        scoresneg.columns = list(range(scoresneg.shape[1]))
    scorecol = 1 if modeltype == "svm" else 5
    seqnamecol = 0 if modeltype == "svm" else 1
    scorespos = scorespos[[seqnamecol, scorecol]]
    scorespos.columns = ["SEQNAME", "SCORE"]
    scoresneg = scoresneg[[seqnamecol, scorecol]]
    scoresneg.columns = ["SEQNAME", "SCORE"]
    if modeltype == "pwm":  # return best score for each sequence
        scorespos = pd.DataFrame(scorespos.groupby("SEQNAME")["SCORE"].max())
        scorespos.reset_index(inplace=True)
        scoresneg = pd.DataFrame(scoresneg.groupby("SEQNAME")["SCORE"].max())
        scoresneg.reset_index(inplace=True)
    # assign labels to datasets sequences
    scorespos["PEAK"] = 1
    scoresneg["PEAK"] = 0
    return pd.concat([scorespos, scoresneg])


def gkmpredict(testpos: str, testneg: str, model: str, outdir: str) -> pd.DataFrame:
    if not os.path.isfile(testpos):
        raise FileNotFoundError(f"{testpos} cannot be found")
    if not os.path.isfile(testneg):
        raise FileNotFoundError(f"{testneg} cannot be found")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"{model} cannot be found")
    scores_prefix = os.path.basename(model).replace(".model.txt", "")
    scorespos = os.path.join(outdir, f"{scores_prefix}.scores.pos.txt")  # positive sequences scores
    scoresneg = os.path.join(outdir, f"{scores_prefix}.scores.neg.txt")  # negative sequences scores
    try:
        subprocess.call(f"gkmpredict -T 16 {testpos} {model} {scorespos}", shell=True)  # score positive sequences
        subprocess.call(f"gkmpredict -T 16 {testneg} {model} {scoresneg}", shell=True)  # score negative sequences
    except OSError as e:
        raise OSError("SVM evaluation failed") from e
    return read_scores(scorespos, scoresneg, "svm")


def fimo(testpos: str, testneg: str, model: str, outdir: str) -> pd.DataFrame:
    if not os.path.isfile(testpos):
        raise FileNotFoundError(f"{testpos} cannot be found")
    if not os.path.isfile(testneg):
        raise FileNotFoundError(f"{testneg} cannot be found")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"{model} cannot be found")
    scores_prefix = os.path.basename(os.path.dirname(model))
    scorespos = os.path.join(outdir, f"{scores_prefix}_pos")  # positive sequences scores
    scoresneg = os.path.join(outdir, f"{scores_prefix}_neg")  # negative sequences scores
    try:
        subprocess.call(f"fimo --oc {scorespos} --max-stored-scores 1000000000 --thresh 1 {model} {testpos}", shell=True)  # score positive sequences
        subprocess.call(f"fimo --oc {scoresneg} --max-stored-scores 1000000000 --thresh 1 {model} {testneg}", shell=True)  # score negative sequences
    except OSError as e:
        raise OSError("PWM evaluation failed") from e
    return read_scores(os.path.join(scorespos, "fimo.txt"), os.path.join(scoresneg, "fimo.txt"), "pwm")

def main():
    testpos, testneg, model, modeltype, outdir = sys.argv[1:]
    outdir = os.path.join(outdir, "evaluation")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if modeltype not in ["svm", "pwm"]:
        raise ValueError("Unknown model type")
    if modeltype == "svm":  # SVM model
        X = gkmpredict(testpos, testneg, model, outdir)
    else:  # PWM model
        X = fimo(testpos, testneg, model, outdir)
    print(X.shape)

if __name__ == "__main__":
    main()





