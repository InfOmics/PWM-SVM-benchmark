"""
"""

from typing import Tuple

import pandas as pd

import pybedtools
import subprocess
import sys
import os

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

def read_bed(bedfile: str) -> pybedtools.BedTool:
    if not os.path.isfile(bedfile):
        raise FileNotFoundError(f"{bedfile} cannot be found")
    return pybedtools.BedTool(bedfile)

def subtract(a: pybedtools.BedTool, b: pybedtools.BedTool) -> pybedtools.BedTool:
    c = a.subtract(b, A=True)  #remove entire feature of A if any overlap with B
    return c

def split_train_test(bedfile: str) -> Tuple[str, str]:
    bed = pd.read_csv(bedfile, sep="\t", header=None)  
    # remove non-canonical chromosomes
    bed = bed[bed[0].isin(CHROMS)]
    # chr2 data for test, other for training
    bed_test = bed[bed[0] == "chr2"]
    bed_train = bed[bed[0] != "chr2"]
    assert bed_test.shape[0] + bed_train.shape[0] == bed.shape[0]
    bedfile_train = f"{os.path.splitext(bedfile)[0]}_train.bed"
    bedfile_test = f"{os.path.splitext(bedfile)[0]}_test.bed"
    bed_train.to_csv(bedfile_train, sep="\t", header=False, index=False)
    bed_test.sample(frac=1).to_csv(bedfile_test, sep="\t", header=False, index=False)
    return bedfile_train, bedfile_test


def sortbed(bedfile: str, threshold: int) -> pybedtools.BedTool:
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    bed = bed.sort_values([8, 6], ascending=False)  # sort peaks by q-value and signal
    bedout = f"{os.path.splitext(bedfile)[0]}.tmp.bed"
    threshold = bed.shape[0] if threshold == 0 else threshold
    bed[:threshold].sample(frac=1).to_csv(bedout, sep="\t", header=False, index=False)
    return pybedtools.BedTool(bedout)

def extract_sequences(bed: pybedtools.BedTool, genome: str) -> str:
    if not os.path.isfile(genome):
        raise FileNotFoundError(f"{genome} cannot be found")
    sequences = bed.sequence(fi=genome)
    fastafile = f"{os.path.splitext(os.path.splitext(bed.fn)[0])[0]}.fa"
    try:
        with open(fastafile, mode="w") as outfile:
            outfile.write(open(sequences.seqfn).read())
    except OSError as e:
        raise OSError(f"An error occurred while writing to {fastafile}") from e
    assert os.stat(fastafile).st_size > 0
    return fastafile

def shuffle(sequences: str) -> str:
    sequences_shuffle = f"{os.path.splitext(sequences)[0]}_shuffle.fa"
    try:
        subprocess.run(["fasta-shuffle-letters", sequences, "-kmer", "2","-dna", "-line", "100000", "-seed", "42", sequences_shuffle])
    except OSError as e:
        raise OSError("Sequence shuffle failed")
    return sequences_shuffle


def construct_dataset(positive: str, threshold: int, genome: str):
    # TODO: uncomment and develop the pipeline for negative sequences (dnas-seq + matching GC/length/repeat)
    # bedpos = read_bed(positive)  # read positive data
    # bedneg = read_bed(negative)  # read negative data
    # bedneg_filt = subtract(bedneg, bedpos)  # remove negative features overlapping positive features
    trainpos, testpos = split_train_test(positive)
    trainpos_sort = sortbed(trainpos, threshold)  # sort positive features
    seqspos_train = extract_sequences(trainpos_sort, genome)  # extract train sequences 
    seqspos_test = extract_sequences(pybedtools.BedTool(testpos), genome)  # extract train sequences
    # create suffled background data for train and test datasets
    shuffle_train = shuffle(seqspos_train)
    shuffle_test = shuffle(seqspos_test)


def main():
    positive, negative, genome, threshold = sys.argv[1:]
    construct_dataset(positive, int(threshold), genome)


if __name__ == "__main__":
    main()

