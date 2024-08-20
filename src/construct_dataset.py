"""
"""

from typing import Tuple, Optional, Dict, List

import pandas as pd

import pybedtools
import subprocess
import random
import sys
import os

# set seed
random.seed(1234)

CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

def compute_seqname(chrom: str, start: int, stop: int) -> str:
    """ """

    return f"{chrom}:{start}-{stop}"

def remove_duplicates_bed(bedfile: str, outdir: str) -> str:
    """ """

    # create outfiles
    outbed = f"{os.path.splitext(os.path.basename(bedfile))[0]}_unique.bed"
    outbed = os.path.join(outdir, outbed)
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    seqnamecol = bed.shape[1]  # column containing seqnames
    bed[seqnamecol] = bed.apply(lambda x: compute_seqname(x[0], x[1], x[2]), axis=1)
    bed = bed.loc[bed.groupby(seqnamecol)[6].idxmax()]  # report the peak with max score
    bed.to_csv(outbed, sep="\t", header=False, index=False)
    return outbed

def read_bed(bedfile: str) -> pybedtools.BedTool:
    """ """

    if not os.path.isfile(bedfile):
        raise FileNotFoundError(f"{bedfile} cannot be found")
    return pybedtools.BedTool(bedfile)

def subtract(a: pybedtools.BedTool, b: pybedtools.BedTool) -> pybedtools.BedTool:
    """ """

    c = a.subtract(b, A=True)  # remove entire feature of A if any overlap with B
    return c

def read_fasta(fastafile: str) -> Dict[str, str]:
    """ """

    try:
        with open(fastafile, mode="r") as infile:
            seqnames = [line.strip()[1:] for line in infile if line.startswith(">")]
        with open(fastafile, mode="r") as infile:
            sequences = [line.strip() for line in infile if not line.startswith(">")]
    except IOError as e:
        raise OSError(f"An error occurred while reading {fastafile}") from e
    return {seqname: sequences[i] for i, seqname in enumerate(seqnames)}


def chrom_split(sequences: Dict[str, str]) -> Dict[str, List[str]]:
    """ """

    seqs_chrom = {chrom: [] for chrom in CHROMS}
    for seqname in sequences:
        chrom = seqname.split(":")[0]
        seqs_chrom[chrom].append(seqname)
    return seqs_chrom
    
def compute_background_data(postrain: str, postest: str, negtrain: str, negtest: str):
    """ """

    postrain, postest = read_fasta(postrain), read_fasta(postest)
    negtrain, negtest = read_fasta(negtrain), read_fasta(negtest)
    postrain_chrom, postest_chrom = chrom_split(postrain), chrom_split(postest)
    negtrain_chrom, negtest_chrom = chrom_split(negtrain), chrom_split(negtest)
    background_train, background_test = [], []
    for chrom in postrain_chrom:
        background_train += random.sample(
            negtrain_chrom[chrom], len(postrain_chrom[chrom])
        )
        background_test += random.sample(
            negtest_chrom[chrom], len(postest_chrom[chrom])
        )
    return background_train, background_test


def write_fasta(fastafile: str, outfasta: str, seqnames: Optional[List[str]] = None) -> None:
    """ """

    fasta = read_fasta(fastafile)
    seqnames = list(fasta.keys()) if seqnames is None else seqnames
    with open(outfasta, mode="w") as outfile:
        for seqname in seqnames:
            outfile.write(f">{seqname}\n{fasta[seqname]}\n")    


def split_train_test(bedfile: str) -> Tuple[str, str]:
    """ """

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


def sortbed(bedfile: str, threshold: int, sort: Optional[bool] = True) -> pybedtools.BedTool:
    """ """

    bed = pd.read_csv(bedfile, sep="\t", header=None)
    if sort:
        bed = bed.sort_values([8, 6], ascending=False)  # sort peaks by q-value and signal
    bedout = f"{os.path.splitext(bedfile)[0]}.tmp.bed"
    threshold = bed.shape[0] if threshold == 0 else threshold
    bed[:threshold].sample(frac=1).to_csv(bedout, sep="\t", header=False, index=False)
    return pybedtools.BedTool(bedout)

def extract_sequences(bed: pybedtools.BedTool, genome: str) -> str:
    """ """

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

def shuffle(sequences: str, sequences_shuffle: str) -> str:
    """ """

    try:
        subprocess.run(["fasta-shuffle-letters", sequences, "-kmer", "2","-dna", "-line", "100000", "-seed", "42", sequences_shuffle])
    except OSError as e:
        raise OSError("Sequence shuffle failed")
    return sequences_shuffle

def construct_dataset(positive: str, threshold: int, genome: str, outdir: str, negative: Optional[str] = None):
    """ """

    # retrieve fname prefix
    fname_prefix = os.path.splitext(os.path.basename(positive))[0]
    # remove potential duplicated peaks in the positive sequences
    positive = remove_duplicates_bed(positive, outdir)
    if negative is not None:
        negative_fname = os.path.join(
            outdir, f"{os.path.splitext(os.path.basename(negative))[0]}"
        )
        negative_filt = subtract(read_bed(negative), read_bed(positive))  # remove negative peaks overlapping positive peaks
        # store filtered negative peaks
        negative_fname_filt = f"{negative_fname}_no_overlap.bed"
        with open(negative_fname_filt, mode="w") as outfile:
            outfile.write(str(negative_filt))
    # split train and test datasets
    trainpos, testpos = split_train_test(positive)
    seqpos_train = extract_sequences(sortbed(trainpos, threshold), genome)
    seqpos_test = extract_sequences(sortbed(testpos, threshold), genome)
    # construct background data matching negative and positive chrom location
    if negative is not None:  
        trainneg, testneg = split_train_test(negative_fname_filt)
        # always take the full background dataset
        seqneg_train = extract_sequences(sortbed(trainneg, 0, sort=False), genome)
        seqneg_test = extract_sequences(sortbed(testneg, 0, sort=False), genome)
        bg_train, bg_test = compute_background_data(seqpos_train, seqpos_test, seqneg_train, seqneg_test)
        write_fasta(seqneg_train, os.path.join(outdir, f"{fname_prefix}_neg_dnase_train.fa"), bg_train)
        write_fasta(seqneg_test, os.path.join(outdir, f"{fname_prefix}_neg_dnase_test.fa"), bg_test)
    else:  # shuffle as background
        shuffle_train = shuffle(seqpos_train, os.path.join(outdir, f"{fname_prefix}_neg_shuffle_train.fa"))
        shuffle_test = shuffle(seqpos_test, os.path.join(outdir, f"{fname_prefix}_neg_shuffle_test.fa"))
    write_fasta(seqpos_train, os.path.join(outdir, f"{fname_prefix}_pos_train.fa"))
    write_fasta(seqpos_test, os.path.join(outdir, f"{fname_prefix}_pos_test.fa"))


def main():
    args = sys.argv[1:]
    if len(args) != 4 and len(args) != 5:  
        raise ValueError("Too many/few input arguments")
    positive, genome, threshold, outdir = args[:4]
    negative = None if len(args) == 4 else args[4]
    construct_dataset(positive, int(threshold), genome, outdir, negative)


if __name__ == "__main__":
    main()

