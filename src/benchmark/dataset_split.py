"""
"""

from typing import Tuple, Optional, Dict, List
from glob import glob
from time import time
from tqdm import tqdm

import pandas as pd

import pybedtools
import subprocess
import random
import sys
import os

# set seed for reproducibility
random.seed(1234)

# limit the analysis to consider only data mapped on canonical chromosomes
CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
# train and test data directory names
TRAINDIR = "traindata"
TESTDIR = "testdata"


def parse_commandline(args: List[str]) -> Tuple[List[str], str, str, str]:
    """ """
    if len(args) != 4:
        raise ValueError(f"Too many/few input arguments ({len(args)})")
    posdir, negdir, genome, datadir = (
        args  # recover folders storing positive and negative data
    )
    if not os.path.isdir(posdir):
        raise FileNotFoundError(f"Unable to locate positive data folder {posdir}")
    if not os.path.isdir(negdir):
        raise FileNotFoundError(f"Unable to locate negative data folder {negdir}")
    # retrieve positive and negative BED files
    posbeds, negbeds = glob(os.path.join(posdir, "*.bed")), glob(
        os.path.join(negdir, "*.bed")
    )
    if not posbeds:
        raise FileNotFoundError(
            f"Positive data folder {posdir} does not contain BED files"
        )
    if not negbeds:
        raise FileNotFoundError(
            f"Negative data folder {negdir} does not contain BED files"
        )
    assert len(negbeds) == 1  # assumes that only one negative dataset is provided
    # check genome reference file existence
    if not os.path.isfile(genome):
        raise FileNotFoundError(f"Unable to locate reference genome {genome}")
    # check data directory existance
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"Unable to locate data folder {datadir}")
    return posbeds, negbeds[0], genome, datadir


def construct_dirtree(datadir: str) -> Tuple[str, str]:
    """ """
    # create the directory tree within data folder
    assert os.path.isdir(datadir)
    traindir, testdir = os.path.join(datadir, TRAINDIR), os.path.join(datadir, TESTDIR)
    for d in [traindir, testdir]:  # train and test directories
        if not os.path.isdir(d):  # if not already present, create directory
            os.makedirs(d)
    return traindir, testdir


def compute_seqname(chrom: str, start: int, stop: int) -> str:
    """ """
    return f"{chrom}:{start}-{stop}"


def remove_duplicates(bedfile: str, fname: str, outdir: str) -> str:
    """ """
    bedunique = os.path.join(outdir, f"{fname}_unique.bed")  # unique peaks fname
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    seqnamescol = bed.shape[1]  # column containing seqnames
    # assign seqname to each peak and use the seqnames as keys for duplicate removal
    bed[seqnamescol] = bed.apply(
        lambda x: compute_seqname(x.iloc[0], x.iloc[1], x.iloc[2]), axis=1
    )
    # if duplicate peaks are detected, keep the one with higher enrichment score
    bed = bed.loc[bed.groupby(seqnamescol)[6].idxmax()]
    bed.to_csv(bedunique, sep="\t", header=None, index=False)
    return bedunique


def read_bed(bedfile: str) -> pybedtools.BedTool:
    """ """
    if not os.path.isfile(bedfile):
        raise FileNotFoundError(f"{bedfile} cannot be found")
    return pybedtools.BedTool(bedfile)


def subtract(a: pybedtools.BedTool, b: pybedtools.BedTool) -> pybedtools.BedTool:
    """ """
    c = a.subtract(b, A=True)  # remove entire feature of A if any overlap with B
    return c


def filter_negative(positive: str, negative: str, fname: str, trainnegdir: str) -> str:
    # remive features in negative dataset even if partially overlapping with
    # features in positive dataset
    negative_filt_bed = subtract(read_bed(negative), read_bed(positive))
    negative_filt_fname = os.path.join(trainnegdir, f"{fname}_neg_no_overlap.bed")
    with open(negative_filt_fname, mode="w") as outfile:
        outfile.write(str(negative_filt_bed))
    return negative_filt_fname


def split_train_test(
    bedfile: str, testchrom: str, fname: str, traindir: str, testdir: str
) -> Tuple[str, str]:
    """ """
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    bed = bed[bed[0].isin(CHROMS)]  # remove data mapped in non canonical chroms
    # split dataset
    bed_train, bed_test = bed[bed[0] != testchrom], bed[bed[0] == testchrom]
    assert (bed_train.shape[0] + bed_test.shape[0]) == bed.shape[0]
    bed_train_fname = os.path.join(traindir, f"{fname}_train.bed")
    bed_train.to_csv(bed_train_fname, sep="\t", header=False, index=False)
    bed_test_fname = os.path.join(testdir, f"{fname}_test.bed")
    bed_test.to_csv(bed_test_fname, sep="\t", header=False, index=False)
    subprocess.call(f"rm {bedfile}", shell=True)  # remove old bed
    return bed_train_fname, bed_test_fname


def shuffle_sequences(
    trainpos: str, testpos: str, trainnegdir: str, testnegdir: str, prefix: str
):
    """ """
    # assigne traun and test dataset file names
    trainneg = os.path.join(trainnegdir, f"{prefix}_shuffle_neg_train.fa")
    testneg = os.path.join(testnegdir, f"{prefix}_shuffle_neg_test.fa")
    # shuffle train positive sequences to recover synthetic train background data
    code = subprocess.call(
        f"fasta-shuffle-letters {trainpos} -kmer 2 -dna -line 100000 -seed 42 {trainneg}",
        shell=True,
    )
    if code != 0:
        raise subprocess.SubprocessError(
            f"Shuffling train positive sequences failed on {os.path.basename(prefix)}"
        )
    # shuffle test positive sequences to recover synthetic test background data
    code = subprocess.call(
        f"fasta-shuffle-letters {testpos} -kmer 2 -dna -line 100000 -seed 42 {testneg}",
        shell=True,
    )
    if code != 0:
        raise subprocess.SubprocessError(
            f"Shuffling test positive sequences failed on {os.path.basename(prefix)}"
        )


def extract_sequences(bedfile: str, genome: str) -> str:
    """ """
    bed = pybedtools.BedTool(bedfile)  # load bedtool object
    sequences = bed.sequence(fi=genome)  # extract sequences from reference
    fasta = f"{os.path.splitext(bed.fn)[0]}.fa"
    with open(fasta, mode="w") as outfile:
        outfile.write(open(sequences.seqfn).read())
    assert os.path.isfile(fasta)
    return fasta


def chrom_split(fname: str) -> Dict[str, List[str]]:
    """ """
    chrom_dict = {c: [] for c in CHROMS}
    with open(fname, mode="r") as infile:
        for line in infile:
            chrom = line.strip().split()[0]
            chrom_dict[chrom].append(line)
    return chrom_dict


def compute_background_data(
    trainpos: str,
    trainneg: str,
    testpos: str,
    testneg: str,
    testchrom: str,
    fname: str,
    trainnegdir: str,
    testnegdir: str,
) -> Tuple[str, str]:
    """ """
    # divide genomic features by chromosome
    trainpos_chrom, testpos_chrom = chrom_split(trainpos), chrom_split(testpos)
    trainneg_chrom, testneg_chrom = chrom_split(trainneg), chrom_split(testneg)
    bgtrain, bgtest = [], []
    for chrom in trainpos_chrom:  # select train features
        bgtrain.extend(random.sample(trainneg_chrom[chrom], len(trainpos_chrom[chrom])))
    # select test features
    test_th = len(testpos_chrom[testchrom]) * 10
    test_th = (
        test_th
        if len(testneg_chrom[testchrom]) > test_th
        else len(testneg_chrom[testchrom])
    )
    bgtest.extend(random.sample(testneg_chrom[testchrom], test_th))
    trainneg_fname = os.path.join(trainnegdir, f"{fname}_neg_train.bed")
    with open(trainneg_fname, mode="w") as outfile:
        outfile.write("".join(bgtrain))
    testneg_fname = os.path.join(testnegdir, f"{fname}_neg_test.bed")
    with open(testneg_fname, mode="w") as outfile:
        outfile.write("".join(bgtest))
    # remove old bed files
    for f in [trainneg, testneg]:
        subprocess.call(f"rm {f}", shell=True)
    return trainneg_fname, testneg_fname


def split_dataset(
    positive: str,
    negative: str,
    genome: str,
    traindir: str,
    testdir: str,
    shuffle: bool,
):
    """ """
    # retrieve experiment basename
    chip_fname = os.path.splitext(os.path.basename(positive))[0]
    trainposdir = (
        os.path.join(traindir, "shuffle/positive")
        if shuffle
        else os.path.join(traindir, "dnase/positive")
    )
    trainnegdir = (
        os.path.join(traindir, "shuffle/negative")
        if shuffle
        else os.path.join(traindir, "dnase/negative")
    )
    testposdir = (
        os.path.join(testdir, "shuffle/positive")
        if shuffle
        else os.path.join(testdir, "dnase/positive")
    )
    testnegdir = (
        os.path.join(testdir, "shuffle/negative")
        if shuffle
        else os.path.join(testdir, "dnase/negative")
    )
    # create train and test data folders
    for d in [trainposdir, trainnegdir, testposdir, testnegdir]:
        if not os.path.isdir(d):
            os.makedirs(d)
    # remove potential duplicate peaks from positive sequences
    positive = remove_duplicates(positive, chip_fname, trainposdir)
    if not shuffle:  # processing dnase peaks (real biological background)
        # remove features from negative overlapping with features on positive
        negative = filter_negative(positive, negative, chip_fname, trainnegdir)
    # split positive dataset in train and test
    trainpos, testpos = split_train_test(
        positive, "chr2", chip_fname, trainposdir, testposdir
    )
    trainpos_seqs, testpos_seqs = extract_sequences(
        trainpos, genome
    ), extract_sequences(testpos, genome)
    if (
        shuffle
    ):  # shuffle input positive sequences (synthetic background) peaks processing
        shuffle_sequences(
            trainpos_seqs, testpos_seqs, trainnegdir, testnegdir, chip_fname
        )
    else:  # dnase peaks processing (real biological background)
        trainneg, testneg = split_train_test(
            negative, "chr2", chip_fname, trainnegdir, testnegdir
        )
        trainneg, testneg = compute_background_data(
            trainpos,
            trainneg,
            testpos,
            testneg,
            "chr2",
            chip_fname,
            trainnegdir,
            testnegdir,
        )
        extract_sequences(trainneg, genome), extract_sequences(testneg, genome)
    # remove old bed files
    fnames = [trainpos, testpos] if shuffle else [trainpos, testpos, trainneg, testneg]
    for f in fnames:
        subprocess.call(f"rm {f}", shell=True)


def main():
    # parse command line arguments -> expected input: folder containing positive
    # sequence dataset (BED format) and folder containing negative sequence
    # dataset (BED format)
    posbeds, negbed, genome, datadir = parse_commandline(sys.argv[1:])
    # construct train and test data directory tree
    traindir, testdir = construct_dirtree(datadir)
    # split each dataset on train and test
    sys.stdout.write("Train and test datasets construction\n")
    start = time()
    for posbed in tqdm(posbeds):
        # shuffle as background (synthetic data)
        split_dataset(posbed, negbed, genome, traindir, testdir, True)
        # dnase as background (real biological background)
        split_dataset(posbed, negbed, genome, traindir, testdir, False)


if __name__ == "__main__":
    main()
