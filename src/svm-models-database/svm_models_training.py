"""
"""

from typing import Tuple, List
from glob import glob
from tqdm import tqdm
from time import time

import subprocess
import sys
import os

GKMTRAIN = "gkmtrain"  # lsgkm train command


def parse_commandline(args: List[str]) -> Tuple[List[str], List[str]]:
    trainposdir, trainnegdir = args
    # check positive and negative training directories consistency
    if not os.path.isdir(trainposdir):
        raise FileNotFoundError(f"Unable to locate positive data folder {trainposdir}")
    if not os.path.isdir(trainnegdir):
        raise FileNotFoundError(f"Unable to locate negative data folder {trainnegdir}")
    return trainposdir, trainnegdir


def train(positive: str, negative: str, modelname: str, modelsdir: str):
    model = os.path.join(modelsdir, f"{modelname}")
    code = subprocess.call(
        f"{GKMTRAIN} -T 16 -v 0 {positive} {negative} {model}", shell=True
    )
    if code != 0:
        raise subprocess.SubprocessError(f"SVM training failed on {positive}")


def train_models(trainposdir: str, trainnegdir: str, modelsdir: str):
    # retrieve positive and negative training data
    posfasta, negfasta = glob(os.path.join(trainposdir, "*.fa")), glob(
        os.path.join(trainnegdir, "*.fa")
    )
    assert len(posfasta) == len(negfasta)
    experiment_names = [os.path.basename(e).split("_")[0] for e in posfasta]
    sys.stdout.write("Training SVM-based motif models\n")
    start = time()
    for experiment_name in tqdm(experiment_names):  # train svm model for each datset
        positive = os.path.join(trainposdir, f"{experiment_name}_train.fa")
        if not os.path.isfile(positive):
            raise FileNotFoundError(
                f"Unable to locate positive training dataset {positive}"
            )
        negative = os.path.join(trainnegdir, f"{experiment_name}_neg_train.fa")
        if not os.path.isfile(negative):
            raise FileNotFoundError(
                f"Unable to locate negative training dataset {negative}"
            )
        train(positive, negative, experiment_name, modelsdir)
    sys.stdout.write(
        f"Training SVM-based motif models completed in {(time() - start):.2f}s\n\n"
    )


def main():
    # parse command line arguments
    trainposdir, trainnegdir = parse_commandline(sys.argv[1:])
    modelsdir = "models"  # create models directory
    if not os.path.isdir(modelsdir):
        os.mkdir(modelsdir)
    # train svm model for each experiment in the training folder
    train_models(trainposdir, trainnegdir, modelsdir)


if __name__ == "__main__":
    main()
