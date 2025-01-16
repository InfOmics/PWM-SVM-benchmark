"""
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
)
from typing import List, Tuple, Dict
from itertools import product
from glob import glob
from tqdm import tqdm
from time import time

import pandas as pd
import numpy as np

import subprocess
import sys
import os


COMPARISONS = ["size", "width", "kernel"]  # benchmark comparisons performed
SIZES = [500, 1000, 2000, 5000, 10000, 0]  # dataset sizes compared
WIDTHS = [50, 100, 150, 200, 0]  # sequence widths compared
FIMO = "fimo"  # fimo command line call
# parameters used to scan sequences with fimo
FIMOOPTIONS = "--max-stored-scores 1000000000 --verbosity 1 --thresh 1"
REPORTCOLS = ["seqname", "score", "pvalue", "peak"]  # colnames used in the report
PERFCOLS = [
    "EXPERIMENT",
    "PRECISION",
    "RECALL",
    "AUPRC",
    "TPR",
    "FPR",
    "AUROC",
    "F1",
]  # performance report column names


def parse_commandline(args: List[str]) -> Tuple[str, str, str]:
    """ """
    comparison, datadir, benchmarkdir = args
    if comparison not in COMPARISONS:  # check correct comparison
        raise ValueError(f"Forbidden comparison requested ({comparison})")
    if not os.path.isdir(datadir):  # check data directory existence
        raise FileNotFoundError(f"Unable to locate benchmark data folder ({datadir})")
    return comparison, datadir, benchmarkdir


def read_fasta(fasta: str) -> Dict[str, str]:
    """ """
    try:
        with open(fasta, mode="r") as infile:
            seqnames = [line.strip()[1:] for line in infile if line.startswith(">")]
        with open(fasta, mode="r") as infile:
            sequences = [line.strip() for line in infile if not line.startswith(">")]
    except OSError as e:
        raise OSError(f"An error occurred while reading {fasta}") from e
    return {seqname: sequences[i] for i, seqname in enumerate(seqnames)}


def score_pwm(fasta: str, seqname: str, sequence: str, pwm: str) -> Tuple[float, float]:
    """ """
    try:
        with open(f"{fasta}.fa", mode="w") as outfile:  # write the chunk seq to fasta
            outfile.write(f">{seqname}\n{sequence}\n")
    except OSError as e:
        raise OSError(f"Writing chunk sequence failed on {fasta}") from e
    # perform sequence scan for input motif with fimo
    code = subprocess.call(
        f"{FIMO} --oc {fasta} {FIMOOPTIONS} {pwm} {fasta}.fa",
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL,
        shell=True,
    )
    if code != 0:
        raise subprocess.SubprocessError(
            f"An error occurred while scanning sequence {fasta} with fimo"
        )
    x = pd.read_csv(os.path.join(fasta, "fimo.tsv"), sep="\t")  # read the TSV report
    subprocess.call(f"rm -r {fasta} {fasta}.fa", shell=True)  # remove tmp data
    return max(x["score"].to_list()), x.loc[x["score"].idxmax(), "p-value"]


def store_scores_pwm(
    scorespos: Dict[str, float], scoresneg: Dict[str, float], prefix: str
) -> None:
    """ """
    scorespos_df = {colname: [] for colname in REPORTCOLS[:-1]}  # except peak column
    for seqname, (score, pvalue) in scorespos.items():
        scorespos_df["seqname"].append(seqname)
        scorespos_df["score"].append(score)
        scorespos_df["pvalue"].append(pvalue)
    scorespos_df = pd.DataFrame(scorespos_df)  # create dataframe
    scorespos_df[REPORTCOLS[3]] = 1  # assign 1 label to chipseq peaks
    scoresneg_df = {colname: [] for colname in REPORTCOLS[:-1]}  # except peak column
    for seqname, (score, pvalue) in scoresneg.items():
        scoresneg_df["seqname"].append(seqname)
        scoresneg_df["score"].append(score)
        scoresneg_df["pvalue"].append(pvalue)
    scoresneg_df = pd.DataFrame(scoresneg_df)  # create dataframe
    scoresneg_df[REPORTCOLS[3]] = 0  # assign 0 label to dnase peaks
    scores = pd.concat(
        [scorespos_df, scoresneg_df]
    )  # concatenate the individual reports
    # shuffle sequence order to avoid biases while computing performance metrics
    scores.sample(frac=1).to_csv(f"{prefix}.tsv", sep="\t", index=False)


def fimo(testpos: str, testneg: str, pwm: str, prefix: str) -> None:
    """ """
    if os.path.isfile(f"{prefix}.tsv"):  # TODO: remove
        return
    workingdir = os.path.join(prefix)  # retrieve working folder from outfile prefix
    if not os.path.isdir(workingdir):
        os.makedirs(workingdir)
    # to avoid memory issues from fimo preventing it to report the score for all
    # test sequences we split the original test fasta file in chunks and score
    # each chunk individually
    posdata = read_fasta(testpos)  # read positive sequences
    negdata = read_fasta(testneg)  # read negative sequences
    # create dictionaries to store the max score assigned to each sequence
    scorespos = {seqname: None for seqname in posdata}
    scoresneg = {seqname: None for seqname in negdata}
    for seqname, sequence in posdata.items():  # score positive data
        fname = os.path.join(workingdir, seqname.replace(":", "_").replace("-", "_"))
        scorespos[seqname] = score_pwm(fname, seqname, sequence, pwm)
    for seqname, sequence in negdata.items():  # score negative data
        fname = os.path.join(workingdir, seqname.replace(":", "_").replace("-", "_"))
        scoresneg[seqname] = score_pwm(fname, seqname, sequence, pwm)
    store_scores_pwm(scorespos, scoresneg, prefix)  # store scores tsv report
    subprocess.call(f"rm -r {prefix}", shell=True)


def store_scores_svm(scorespos_fname: str, scoresneg_fname: str, prefix: str) -> None:
    """ """
    # load svm scores in pandas dataframes
    scorespos = pd.read_csv(scorespos_fname, header=None, sep="\t")
    scoresneg = pd.read_csv(scoresneg_fname, header=None, sep="\t")
    scorespos.columns = REPORTCOLS[:2]  # initialize scores report colnames
    scoresneg.columns = REPORTCOLS[:2]  # initialize scores report colnames
    scorespos[REPORTCOLS[2]] = 0  # pvalue not provided
    scoresneg[REPORTCOLS[2]] = 0  # pvalue not provided
    scorespos[REPORTCOLS[3]] = 1  # assign 1 label to chipseq peaks
    scoresneg[REPORTCOLS[3]] = 0  # assign 0 label to dnase peaks
    scores = pd.concat([scorespos, scoresneg])
    # shuffle sequence order to avoid biases while computing performance metrics
    scores.sample(frac=1).to_csv(f"{prefix}.tsv", sep="\t", index=False)
    subprocess.call(f"rm {scoresneg_fname} {scorespos_fname}", shell=True)


def gkmpredict(testpos: str, testneg: str, model: str, prefix: str) -> None:
    """ """
    scorespos = f"{prefix}.scores.pos.txt"  # positive sequences scores
    scoresneg = f"{prefix}.scores.neg.txt"  # negative sequences scores
    code = subprocess.call(
        f"gkmpredict -T 16 {testpos} {model} {scorespos}",
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL,
        shell=True,
    )  # score positive sequences
    if code != 0:
        raise subprocess.SubprocessError(
            f"An error occurred while scoring positive sequences {testpos}"
        )
    code = subprocess.call(
        f"gkmpredict -T 16 {testneg} {model} {scoresneg}",
        stderr=subprocess.STDOUT,
        stdout=subprocess.DEVNULL,
        shell=True,
    )  # score negative sequences
    if code != 0:
        raise subprocess.SubprocessError(
            f"An error occurred while scoring negative sequences {testneg}"
        )
    store_scores_svm(scorespos, scoresneg, prefix)
    subprocess.call(f"rm -r {prefix}", shell=True)


def score_models_size(
    modelsdir: str, scoresdir: str, testdatadir_pos: str, testdatadir_neg: str, bg: str
) -> None:
    """ """
    # retrieve experiment names
    experiment_names = {
        os.path.basename(d).split("_")[0]
        for d in glob(os.path.join(modelsdir, "size_500/*_streme"))
    }
    assert len(experiment_names) == 59  # should be 59 experiments
    for size in SIZES:
        sizedir = "size_full" if size == 0 else f"size_{size}"
        modelsdir_size = os.path.join(modelsdir, sizedir)  # models folder
        scoresoutdir = os.path.join(scoresdir, sizedir)  # scores folder
        if not os.path.isdir(scoresoutdir):  # create scores folder if not present
            os.makedirs(scoresoutdir)
        sys.stdout.write(f"size - {sizedir}\n")
        for experiment_name in tqdm(experiment_names):
            # retrieve positive and negative test datasets
            testpos = os.path.join(testdatadir_pos, f"{experiment_name}_test.fa")
            testneg = os.path.join(
                testdatadir_neg, f"{experiment_name}_{bg}_neg_test.fa"
            )
            fimo(
                testpos,
                testneg,
                os.path.join(modelsdir_size, f"{experiment_name}_meme/meme.txt"),
                os.path.join(scoresoutdir, f"{experiment_name}_meme"),
            )  # meme
            # fimo(testpos, testneg, os.path.join(modelsdir_size, f"{experiment_name}_streme/streme.txt"), os.path.join(scoresoutdir, f"{experiment_name}_streme"))  # streme
            # gkmpredict(testpos, testneg, os.path.join(modelsdir_size, f"{experiment_name}_svm/{experiment_name}.model.txt"), os.path.join(scoresoutdir, f"{experiment_name}_svm"))  # svm


def score_models_width(
    modelsdir: str, scoresdir: str, testdatadir_pos: str, testdatadir_neg: str, bg: str
) -> None:
    """ """
    # retrieve experiment names
    experiment_names = {
        os.path.basename(d).split("_")[0]
        for d in glob(os.path.join(modelsdir, "width_50/*_streme"))
    }
    assert len(experiment_names) == 59  # should be 59 experiments
    for width in WIDTHS:
        widthdir = "width_full" if width == 0 else f"width_{width}"
        modelsdir_size = os.path.join(modelsdir, widthdir)  # models folder
        scoresoutdir = os.path.join(scoresdir, widthdir)  # scores folder
        if not os.path.isdir(scoresoutdir):  # create scores folder if not present
            os.makedirs(scoresoutdir)
        sys.stdout.write(f"width - {widthdir}\n")
        for experiment_name in tqdm(experiment_names):
            # retrieve positive and negative test datasets
            testpos = os.path.join(testdatadir_pos, f"{experiment_name}_test.fa")
            testneg = os.path.join(
                testdatadir_neg, f"{experiment_name}_{bg}_neg_test.fa"
            )
            fimo(
                testpos,
                testneg,
                os.path.join(modelsdir_size, f"{experiment_name}_meme/meme.txt"),
                os.path.join(scoresoutdir, f"{experiment_name}_meme"),
            )  # meme
            # fimo(testpos, testneg, os.path.join(modelsdir_size, f"{experiment_name}_streme/streme.txt"), os.path.join(scoresoutdir, f"{experiment_name}_streme"))  # streme
            # gkmpredict(testpos, testneg, os.path.join(modelsdir_size, f"{experiment_name}_svm/{experiment_name}.model.txt"), os.path.join(scoresoutdir, f"{experiment_name}_svm"))  # svm


def score_models(comparison: str, datadir: str, benchmarkdir: str):
    """ """
    if comparison == COMPARISONS[0]:  # compare performance on different dataset sizes
        for bg_model, bg_score in [
            ("shuffle", "dnase-1"),
            ("dnase", "dnase-1"),
        ]:  # list(product(["shuffle", "dnase"], repeat=2)):
            modelsdir = os.path.join(
                benchmarkdir, f"models/dataset-size-comparison/{bg_model}"
            )
            testdatadir_pos = os.path.join(datadir, f"testdata/{bg_score}/positive")
            testdatadir_neg = os.path.join(datadir, f"testdata/{bg_score}/negative")
            scoresdir = os.path.join(
                benchmarkdir, f"scores/dataset-size-comparison/{bg_model}-{bg_score}"
            )
            score_models_size(
                modelsdir,
                scoresdir,
                testdatadir_pos,
                testdatadir_neg,
                bg_score.replace("-1", ""),
            )
    if comparison == COMPARISONS[1]:  # compare performance on sequence width
        for bg_model, bg_score in [
            ("shuffle", "dnase-1"),
            ("dnase", "dnase-1"),
        ]:  # list(product(["shuffle", "dnase"], repeat=2)):
            modelsdir = os.path.join(
                benchmarkdir, f"models/sequence-width-comparison/{bg_model}"
            )
            testdatadir_pos = os.path.join(datadir, f"testdata/{bg_score}/positive")
            testdatadir_neg = os.path.join(datadir, f"testdata/{bg_score}/negative")
            scoresdir = os.path.join(
                benchmarkdir, f"scores/sequence-width-comparison/{bg_model}-{bg_score}"
            )
            score_models_width(
                modelsdir,
                scoresdir,
                testdatadir_pos,
                testdatadir_neg,
                bg_score.replace("-1", ""),
            )


def read_scores(scores_fname: str) -> pd.DataFrame:
    """ """
    return pd.read_csv(scores_fname, sep="\t")


def compute_precision(scores: str, tool: str) -> float:
    """ """
    y_test = scores[REPORTCOLS[3]].tolist()
    y_pred_proba = (
        scores[REPORTCOLS[1]].tolist() if tool == "svm" else scores[REPORTCOLS[2]]
    )
    y_test, y_pred_proba = scores[REPORTCOLS[3]], scores[REPORTCOLS[1]]
    precision, _, thresholds = precision_recall_curve(y_test, y_pred_proba)
    return precision[len(thresholds) // 2]


def compute_recall(scores: str, tool: str) -> float:
    """ """
    y_test = scores[REPORTCOLS[3]].tolist()
    y_pred_proba = (
        scores[REPORTCOLS[1]].tolist() if tool == "svm" else scores[REPORTCOLS[2]]
    )
    y_test, y_pred_proba = scores[REPORTCOLS[3]], scores[REPORTCOLS[1]]
    _, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    return recall[len(thresholds) // 2]


def compute_auprc(scores: pd.DataFrame) -> float:
    """ """
    # retrieve true and predicted labels
    y_test, y_pred_proba = scores[REPORTCOLS[3]], scores[REPORTCOLS[1]]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    return auc(recall, precision)  # compute auprc


def compute_tpr(scores: str, tool: str) -> float:
    """ """
    y_test = scores[REPORTCOLS[3]].tolist()
    y_pred_proba = (
        scores[REPORTCOLS[1]].tolist() if tool == "svm" else scores[REPORTCOLS[2]]
    )
    y_test, y_pred_proba = scores[REPORTCOLS[3]], scores[REPORTCOLS[1]]
    _, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    return tpr[len(thresholds) // 2]


def compute_fpr(scores: str, tool: str) -> float:
    """ """
    y_test = scores[REPORTCOLS[3]].tolist()
    y_pred_proba = (
        scores[REPORTCOLS[1]].tolist() if tool == "svm" else scores[REPORTCOLS[2]]
    )
    y_test, y_pred_proba = scores[REPORTCOLS[3]], scores[REPORTCOLS[1]]
    fpr, _, thresholds = roc_curve(y_test, y_pred_proba)
    return fpr[len(thresholds) // 2]


def compute_auroc(scores: pd.DataFrame) -> float:
    """ """
    # retrieve true and predicted labels
    y_test, y_pred_proba = scores[REPORTCOLS[3]], scores[REPORTCOLS[1]]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return auc(fpr, tpr)  # compute auroc


def compute_f1(scores: pd.DataFrame, tool: str) -> float:
    """ """
    y_test = scores[REPORTCOLS[3]].tolist()
    y_pred_proba = (
        scores[REPORTCOLS[1]].tolist() if tool == "svm" else scores[REPORTCOLS[2]]
    )
    # assign labels to predictions
    if tool == "svm":
        y_pred = [1 if l > 0 else 0 for l in y_pred_proba]
    else:  # on pwm check pvalues
        y_pred = [1 if l < 1e-4 else 0 for l in y_pred_proba]
    return f1_score(y_test, y_pred)


def evaluate_models_size(scoresdir: str, perfdir: str) -> None:
    """ """
    # retrieve experiment names
    experiment_names = {
        os.path.basename(d).split("_")[0]
        for d in glob(os.path.join(scoresdir, "size_500/*_streme.tsv"))
    }
    assert len(experiment_names) == 59
    for size in SIZES:
        sizedir = sizedir = "size_full" if size == 0 else f"size_{size}"
        sys.stdout.write(f"size - {sizedir}\n")
        scoresdir_size = os.path.join(scoresdir, sizedir)
        # initialize models performance report
        report = {cname: [] for cname in PERFCOLS}
        for tool in ["meme"]:
            perftable = os.path.join(perfdir, f"summary_table_{sizedir}_{tool}.tsv")
            for experiment_name in tqdm(
                experiment_names
            ):  # iterate over all models' scores
                scores = read_scores(
                    os.path.join(scoresdir_size, f"{experiment_name}_{tool}.tsv")
                )
                report[PERFCOLS[0]].append(experiment_name)
                report[PERFCOLS[1]].append(compute_precision(scores, tool))  # precision
                report[PERFCOLS[2]].append(compute_recall(scores, tool))  # recall
                report[PERFCOLS[3]].append(compute_auprc(scores))  # auprc
                report[PERFCOLS[4]].append(compute_tpr(scores, tool))  # tpr
                report[PERFCOLS[5]].append(compute_fpr(scores, tool))  # fpr
                report[PERFCOLS[6]].append(compute_auroc(scores))  # auroc
                report[PERFCOLS[7]].append(compute_f1(scores, tool))  # f1
            pd.DataFrame(report).to_csv(perftable, sep="\t", index=False)


def evaluate_models_width(scoresdir: str, perfdir: str) -> None:
    """ """
    # retrieve experiment names
    experiment_names = {
        os.path.basename(d).split("_")[0]
        for d in glob(os.path.join(scoresdir, "width_50/*_streme.tsv"))
    }
    assert len(experiment_names) == 59
    for width in WIDTHS:
        widthdir = "width_full" if width == 0 else f"width_{width}"
        sys.stdout.write(f"width - {widthdir}\n")
        scoresdir_size = os.path.join(scoresdir, widthdir)
        # initialize models performance report
        report = {cname: [] for cname in PERFCOLS}
        for tool in ["meme"]:
            perftable = os.path.join(perfdir, f"summary_table_{widthdir}_{tool}.tsv")
            for experiment_name in tqdm(
                experiment_names
            ):  # iterate over all models' scores
                scores = read_scores(
                    os.path.join(scoresdir_size, f"{experiment_name}_{tool}.tsv")
                )
                report[PERFCOLS[0]].append(experiment_name)
                report[PERFCOLS[1]].append(compute_precision(scores, tool))  # precision
                report[PERFCOLS[2]].append(compute_recall(scores, tool))  # recall
                report[PERFCOLS[3]].append(compute_auprc(scores))  # auprc
                report[PERFCOLS[4]].append(compute_tpr(scores, tool))  # tpr
                report[PERFCOLS[5]].append(compute_fpr(scores, tool))  # fpr
                report[PERFCOLS[6]].append(compute_auroc(scores))  # auroc
                report[PERFCOLS[7]].append(compute_f1(scores, tool))  # f1
            pd.DataFrame(report).to_csv(perftable, sep="\t", index=False)


def evaluate_models_performance(comparison: str, benchmarkdir: str):
    """ """
    if comparison == COMPARISONS[0]:  # compare performance on different dataset sizes
        for bg_model, bg_score in [
            ("shuffle", "dnase-1"),
            ("dnase", "dnase-1"),
        ]:  # list(product(["shuffle", "dnase"], repeat=2)):
            scoresdir = os.path.join(
                benchmarkdir, f"scores/dataset-size-comparison/{bg_model}-{bg_score}"
            )
            perfdir = os.path.join(
                benchmarkdir,
                f"performance/dataset-size-comparison/{bg_model}-{bg_score}",
            )
            # if not already present create performance folder
            if not os.path.isdir(perfdir):
                os.makedirs(perfdir)
            evaluate_models_size(scoresdir, perfdir)
    if comparison == COMPARISONS[1]:  # compare performance on different sequence widths
        for bg_model, bg_score in [
            ("shuffle", "dnase-1"),
            ("dnase", "dnase-1"),
        ]:  # list(product(["shuffle", "dnase"], repeat=2)):
            scoresdir = os.path.join(
                benchmarkdir, f"scores/sequence-width-comparison/{bg_model}-{bg_score}"
            )
            perfdir = os.path.join(
                benchmarkdir,
                f"performance/sequence-width-comparison/{bg_model}-{bg_score}",
            )
            # if not already present create performance folder
            if not os.path.isdir(perfdir):
                os.makedirs(perfdir)
            evaluate_models_width(scoresdir, perfdir)


def main():
    # parse input arguments from command line
    comparison, datadir, benchmarkdir = parse_commandline(sys.argv[1:])
    # compute scores for each model
    score_models(comparison, datadir, benchmarkdir)
    evaluate_models_performance(comparison, benchmarkdir)


if __name__ == "__main__":
    main()
