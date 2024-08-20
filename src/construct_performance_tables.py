""" 
"""

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from typing import List, Tuple
from glob import glob

import pandas as pd

import sys
import os

COLNAMES = ["EXPERIMENT", "AUPRC", "AUROC"]
METHODS = ["all", "svm", "meme", "streme"]

def parse_commandline(args: List[str]) -> Tuple[str, str, str, str]:
    if len(args) != 4:
        raise ValueError(f"Too many/few input arguments")
    scoresdir, label, method, outdir = args
    if not os.path.isdir(scoresdir):
        raise FileNotFoundError(f"Cannot find {scoresdir}")
    if method not in METHODS:
        raise ValueError(f"Forbidden method selected ({method})")
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f"Cannot find {outdir}")
    return scoresdir, label, method, outdir

def read_scores(score_fname: str) -> pd.DataFrame:
    return pd.read_csv(score_fname, sep="\t")

def compute_auprc(scores: pd.DataFrame) -> float:
    if any(c not in scores.columns.tolist() for c in ["SCORE", "PEAK"]):
        raise ValueError("SCORE or PEAK not in columns")
    y_test, y_pred_proba = scores["PEAK"], scores["SCORE"]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    return auc(recall, precision)

def compute_auroc(scores: pd.DataFrame) -> float:
    if any(c not in scores.columns.tolist() for c in ["SCORE", "PEAK"]):
        raise ValueError("SCORE or PEAK not in columns")
    y_test, y_pred_proba = scores["PEAK"], scores["SCORE"]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return auc(fpr, tpr)

def summary_by_method(scoresdir: str, method: str, label: str):
    fnames = glob(os.path.join(scoresdir, f"*_{method}.tsv"))
    colnames = COLNAMES[:1] + [f"{c}_{method}_{label}" for c in COLNAMES[1:]]
    table = {c: [] for c in colnames}
    for f in fnames:
        scores = read_scores(f)  # read scores
        table[colnames[0]].append(os.path.basename(f).split("_")[0])
        table[colnames[1]].append(compute_auprc(scores))
        table[colnames[2]].append(compute_auroc(scores))
    return pd.DataFrame(table)

def construct_summary_report(scoresdir: str, method: str, label: str, outdir: str):
    tables = {m: None for m in METHODS[1:]}
    if method == METHODS[0] or method == METHODS[1]:  # svm scores
        tables[METHODS[1]] = summary_by_method(scoresdir, METHODS[1], label)
    if method == METHODS[0] or method == METHODS[2]:  # meme scores
        tables[METHODS[2]] = summary_by_method(scoresdir, METHODS[2], label)
    if method == METHODS[0] or method == METHODS[3]:  # streme scores
        tables[METHODS[3]] = summary_by_method(scoresdir, METHODS[3], label)
    if method == METHODS[0]:  # all
        report = tables[METHODS[1]].merge(tables[METHODS[2]], on=COLNAMES[0], how="outer")
        report = report.merge(tables[METHODS[3]], on=COLNAMES[0], how="outer")
    else:
        report = tables[method]
    report.to_csv(
        os.path.join(outdir, f"summary_table_{label}.tsv"), sep="\t", index=False
    )

def main():
    scoresdir, label, method, outdir = parse_commandline(sys.argv[1:])
    construct_summary_report(scoresdir, method, label, outdir)

if __name__ == "__main__":
    main()