"""
"""

from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from typing import List, Tuple
from glob import glob
from tqdm import tqdm
from time import time

import pandas as pd

import subprocess
import sys
import os

SCORESDIR = "scores"  # folder storing score files
GKMPREDICT = "gkmpredict"  # lsgkm scoring functionality
REPORT = "models_report.csv"  # models performance report
REPORTCOLNAMES = ["EXPERIMENT", "AUPRC", "AUROC", "F1"]  # report column names


def parse_commandline(args: List[str]) -> Tuple[List[str], List[str], List[str]]:
    if len(args) != 3:
        raise ValueError(f"Too many/few input arguments ({len(args)})")
    testposdir, testnegdir, modelsdir = args
    if not os.path.isdir(testposdir):
        raise FileNotFoundError(
            f"Unable to locate positive test data folder {testposdir}"
        )
    if not os.path.isdir(testnegdir):
        raise FileNotFoundError(
            f"Unable to locate negative test data folder {testnegdir}"
        )
    if not os.path.isdir(modelsdir):
        raise FileNotFoundError(f"Unable to locate models folder {modelsdir}")
    return testposdir, testnegdir, modelsdir


def create_scoredir() -> str:
    if not os.path.isdir(SCORESDIR):
        os.mkdir(SCORESDIR)
    return SCORESDIR


def score(testpos: str, testneg: str, model: str, scores_fname: str) -> None:
    scorespos = f"{scores_fname}.scores.pos.txt"  # score positive sequences
    code = subprocess.call(
        f"{GKMPREDICT} -T 16 -v 0 {testpos} {model} {scorespos}", shell=True
    )
    if code != 0:
        raise subprocess.SubprocessError(f"An error occurred while scoring {testpos}")
    scoresneg = f"{scores_fname}.scores.neg.txt"  # score negative sequences
    code = subprocess.call(
        f"{GKMPREDICT} -T 16 -v 0 {testneg} {model} {scoresneg}", shell=True
    )
    if code != 0:
        raise subprocess.SubprocessError(f"An error occurred while scoring {testneg}")


def score_models(testposdir: str, testnegdir: str, modelsdir: str, scoresdir: str):
    # retrieve experiment names
    experiment_names = [
        os.path.basename(m).split(".")[0]
        for m in glob(os.path.join(modelsdir, "*.model.txt"))
    ]
    sys.stdout.write("Scoring test sequences with SVM-based motif models\n")
    start = time()
    for experiment_name in tqdm(experiment_names):  # score each trained model
        model = os.path.join(modelsdir, f"{experiment_name}.model.txt")
        testpos = os.path.join(testposdir, f"{experiment_name}_test.fa")
        testneg = os.path.join(testnegdir, f"{experiment_name}_neg_test.fa")
        scores_fname = os.path.join(scoresdir, experiment_name)
        if os.stat(testpos).st_size <= 0:  # ignore experiments without test data
            continue
        score(testpos, testneg, model, scores_fname)
    sys.stdout.write(
        f"Scoring test sequences with SVM-based motif models completed in {(time() - start):.2f}s\n\n"
    )


def read_scores(scores_fname: str, pos: bool) -> pd.DataFrame:
    scores = pd.read_csv(scores_fname, sep="\t", header=None)
    scores[scores.shape[1]] = 1 if pos else 0  # 1 if peak, 0 otherwise
    return scores


def compute_auprc(scores: pd.DataFrame) -> float:
    y_test, y_pred_proba = scores[2], scores[1]  # retrieve true and predicted labels
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    return auc(recall, precision)  # compute auprc


def compute_auroc(scores: pd.DataFrame) -> float:
    y_test, y_pred_proba = scores[2], scores[1]  # retrieve true and predicted labels
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return auc(fpr, tpr)  # compute auroc


def compute_f1(scores: pd.DataFrame) -> float:
    y_test, y_pred_proba = scores[2].tolist(), scores[1].tolist()
    y_pred = [1 if l > 0 else 0 for l in y_pred_proba]  # assign labels to predictions
    return f1_score(y_test, y_pred)


def evaluate_models_performance(scoresdir: str):
    # retrieve experiment names
    experiment_names = set(
        [
            os.path.basename(s).split(".")[0]
            for s in glob(os.path.join(scoresdir, "*.scores.*.txt"))
        ]
    )
    # initialize models performance report
    report = {cname: [] for cname in REPORTCOLNAMES}
    sys.stdout.write("Evaluating SVM-based motif models\n")
    start = time()
    for experiment_name in tqdm(experiment_names):  # iterate over all models' scores
        # read individual positive and negative scores data
        scorespos = read_scores(
            os.path.join(scoresdir, f"{experiment_name}.scores.pos.txt"), True
        )
        scoresneg = read_scores(
            os.path.join(scoresdir, f"{experiment_name}.scores.neg.txt"), False
        )
        scores = pd.concat([scorespos, scoresneg])  # concatenate pos and neg scores
        report[REPORTCOLNAMES[0]].append(experiment_name)
        report[REPORTCOLNAMES[1]].append(compute_auprc(scores))
        report[REPORTCOLNAMES[2]].append(compute_auroc(scores))
        report[REPORTCOLNAMES[3]].append(compute_f1(scores))
    sys.stdout.write(
        f"Evaulating SVM-based motif models completed in {(time() - start):.2f}s\n\n"
    )
    # store models performance report
    report = pd.DataFrame(report)
    report.to_csv(REPORT, index=False)
    subprocess.call(f"rm -rf {scoresdir}", shell=True)  # remove scores folder


def main():
    # parse command line arguments
    testposdir, testnegdir, modelsdir = parse_commandline(sys.argv[1:])
    scoresdir = create_scoredir()  # create scores folder
    score_models(
        testposdir, testnegdir, modelsdir, scoresdir
    )  # score each trained model
    # construct performance report for each trained model
    evaluate_models_performance(scoresdir)


if __name__ == "__main__":
    main()
