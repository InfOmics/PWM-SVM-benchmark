"""
"""

from typing import List, Tuple, Set
from glob import glob
from tqdm import tqdm

import subprocess
import sys
import os

# computational tools used to train motif models
TOOLS = ["lsgkm", "meme", "streme"]
# default parameters for meme and streme (pwm models)
MEMEDEFAULT = "-dna -mod zoops -nmotifs 1 -minw 6 -maxw 30 -revcomp"
STREMEDEFAULT = "--objfun de --seed 42 --dna --nmotifs 1 --minw 8 --maxw 15"
COMPARISONS = ["size", "width", "kernel"]  # benchmark comparisons performed
SIZES = [500, 1000, 2000, 5000, 10000, 0]  # dataset sizes compared
WIDTHS = [50, 100, 150, 200, 0]  # sequence widths compared


def parse_commandline(args: List[str]) -> Tuple[str, str, str]:
    """ """
    comparison, benchdatadir, benchmarkdir = args
    if comparison not in COMPARISONS:  # check correct comparison
        raise ValueError(f"Forbidden comparison requested ({comparison})")
    if not os.path.isdir(benchdatadir):  # check benchmark data directory existence
        raise FileNotFoundError(
            f"Unable to locate benchmark data folder ({benchdatadir})"
        )
    return comparison, benchdatadir, benchmarkdir


def retrieve_experiment_names(datadir: str) -> Set[str]:
    """ """
    # retrieve positive and negative sequences fasta
    positives = glob(os.path.join(datadir, "*.fa"))
    return {os.path.basename(f).split("_")[0] for f in positives}


def meme(sequences: str, outprefix: str) -> str:
    """ """
    outdir = f"{outprefix}_meme"  # output directory for meme model
    try:
        code = subprocess.call(
            f"meme -oc {outdir} {MEMEDEFAULT} {sequences}",
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.DEVNULL,
        )
        if code != 0:
            raise subprocess.SubprocessError(f"MEME PWM training failed on {sequences}")
    except OSError as e:
        raise OSError("MEME training failed") from e


def streme(positive: str, negative: str, outprefix: str) -> str:
    """ """
    outdir = f"{outprefix}_streme"  # output directory for streme model
    try:
        code = subprocess.call(
            f"streme -oc {outdir} {STREMEDEFAULT} --n {negative} --p {positive}",
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.DEVNULL,
        )
        if code != 0:
            raise subprocess.SubprocessError(
                f"STREME PWM training failed on {positive}"
            )
    except OSError as e:
        raise OSError("STREME training failed") from e


def gkmtrain(positive: str, negative: str, kernel: int, outprefix: str) -> str:
    """ """
    # compute svm model using lsgkm
    outdir = f"{outprefix}_svm"  # output directory for lsgkm
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    modelprefix = os.path.join(outdir, os.path.basename(outprefix))
    # RBF kernels (3 and 5) work best with -c 10 -g 2
    options = "-c 10 -g 2" if kernel in [3, 5] else ""
    try:  # run lsgkm
        code = subprocess.call(
            f"gkmtrain -t {kernel} {options} -T 16 {positive} {negative} {modelprefix}",
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.DEVNULL,
        )
        if code != 0:
            raise subprocess.SubprocessError(f"SVM training failed on {positive}")
    except OSError as e:
        raise OSError("SVM training failed") from e


def train_models_size(benchdatadir: str, bg: str, modelsdir: str):
    """ """
    # retrieve experiment names
    experiment_names = retrieve_experiment_names(
        os.path.join(benchdatadir, f"{bg}/train/fasta/positive/size_500")
    )
    for size in SIZES:  # iterate over dataset sizes
        sizedir = "size_full" if size == 0 else f"size_{size}"
        sys.stdout.write(f"size - {sizedir}\n")
        trainposdir = os.path.join(benchdatadir, f"{bg}/train/fasta/positive/", sizedir)
        trainnegdir = os.path.join(benchdatadir, f"{bg}/train/fasta/negative/", sizedir)
        modelsoutdir = os.path.join(modelsdir, bg, sizedir)
        if not os.path.isdir(modelsoutdir):
            os.makedirs(modelsoutdir)
        for experiment_name in tqdm(experiment_names):
            positive = os.path.join(trainposdir, f"{experiment_name}_train.fa")
            negative = os.path.join(trainnegdir, f"{experiment_name}_{bg}_neg_train.fa")
            # meme(positive, os.path.join(modelsoutdir, experiment_name))  # meme train
            streme(
                positive, negative, os.path.join(modelsoutdir, experiment_name)
            )  # streme train
            # gkmtrain(positive, negative, 4, os.path.join(modelsoutdir, experiment_name))  # lsgkm train


def train_models_width(benchdatadir: str, bg: str, modelsdir: str):
    """ """
    # retrieve experiment names
    experiment_names = retrieve_experiment_names(
        os.path.join(benchdatadir, f"{bg}/train/fasta/positive/width_50")
    )
    for width in WIDTHS:  # iterate over dataset sizes
        widthdir = "width_full" if width == 0 else f"width_{width}"
        sys.stdout.write(f"width - {widthdir}\n")
        trainposdir = os.path.join(
            benchdatadir, f"{bg}/train/fasta/positive/", widthdir
        )
        trainnegdir = os.path.join(
            benchdatadir, f"{bg}/train/fasta/negative/", widthdir
        )
        modelsoutdir = os.path.join(modelsdir, bg, widthdir)
        if not os.path.isdir(modelsoutdir):
            os.makedirs(modelsoutdir)
        for experiment_name in tqdm(experiment_names):
            positive = os.path.join(trainposdir, f"{experiment_name}_train.fa")
            negative = os.path.join(trainnegdir, f"{experiment_name}_{bg}_neg_train.fa")
            meme(positive, os.path.join(modelsoutdir, experiment_name))  # meme train
            # streme(positive, negative, os.path.join(modelsoutdir, experiment_name))  # streme train
            # gkmtrain(positive, negative, 4, os.path.join(modelsoutdir, experiment_name))  # lsgkm train


def train_models(comparison: str, benchdatadir: str, benchmarkdir: str):
    """ """
    if comparison == COMPARISONS[0]:  # compare performance on different dataset sizes
        benchdatadir = os.path.join(benchdatadir, "dataset-size-comparison")
        modelsdir = os.path.join(benchmarkdir, "models/dataset-size-comparison")
        train_models_size(benchdatadir, "shuffle", modelsdir)  # sythentic bg
        train_models_size(benchdatadir, "dnase", modelsdir)  # real bg
    elif (
        comparison == COMPARISONS[1]
    ):  # compare performance on different sequence widths
        benchdatadir = os.path.join(benchdatadir, "sequence-width-comparison")
        modelsdir = os.path.join(benchmarkdir, "models/sequence-width-comparison")
        train_models_width(benchdatadir, "shuffle", modelsdir)  # sythentic bg
        train_models_width(benchdatadir, "dnase", modelsdir)  # real bg


def main():
    # parse input arguments from command line
    comparison, benchdatadir, benchmarkdir = parse_commandline(sys.argv[1:])
    # train motif models
    train_models(comparison, benchdatadir, benchmarkdir)


if __name__ == "__main__":
    main()
