""" 
"""

from typing import List, Tuple, Dict
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np

import pybedtools
import subprocess
import random
import sys
import os


# set seed for reproducibility
random.seed(1234)

# limit the analysis to consider only data mapped on canonical chromosomes
CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
COMPARISONS = ["size", "width", "optimal-global", "optimal-local"]  # benchmark comparisons performed
SIZES = [500, 1000, 2000, 5000, 10000, 0]  # dataset sizes compared
WIDTHS = [50, 100, 150, 200, 0]  # sequence widths compared
TOOLS = ["meme", "streme", "svm"]


def parse_commandline(args: List[str]) -> Tuple[str, str, str]:
    """ """
    if len(args) != 4:
        raise ValueError(f"Too many/few input arguments ({len(args)})")
    # parse input arguments from command line
    comparison, traindatadir, genome, benchdir = args
    if comparison not in COMPARISONS:  # benchmark comparison to perform
        raise ValueError(f"Forbidden comparison requested ({comparison})")
    if not os.path.isdir(traindatadir):  # check existence of train data folder
        raise FileNotFoundError(f"Unable to locate train data folder ({traindatadir})")
    # check genome reference file existence
    if not os.path.isfile(genome):
        raise FileNotFoundError(f"Unable to locate reference genome {genome}")
    return comparison, traindatadir, genome, benchdir


def read_bed(bedfile: str) -> pd.DataFrame:
    """ """
    return pd.read_csv(bedfile, header=None, sep="\t")  # load input bed file


def sort_bed(bed: pd.DataFrame, prefix: str, size: int) -> pybedtools.BedTool:
    """ """
    bed = bed.sort_values([8, 6], ascending=False)  # sort peaks by q-value and enrichment 
    size = bed.shape[0] if size == 0 else size  # 0 for full dataset
    bedout = f"{prefix}_train.bed"
    # to avoid training bias, shuffle the order of peaks
    bed[:size].sample(frac=1).to_csv(bedout, sep="\t", header=False, index=False)
    return pybedtools.BedTool(bedout)


def extract_sequences(bed: pybedtools.BedTool, genome: str) -> str:
    """ """
    sequences = bed.sequence(fi=genome)  # extract sequences from reference
    fasta = f"{os.path.splitext(bed.fn)[0]}.fa"
    with open(fasta, mode="w") as outfile:
        outfile.write(open(sequences.seqfn).read())
    assert os.path.isfile(fasta)
    return fasta


def read_fasta(fasta: str) -> Dict[str, str]:
    """ """
    try:
        with open(fasta, mode="r") as infile:  # recover seqnames
            seqnames = [line.strip()[1:] for line in infile if line.startswith(">")]
        with open(fasta, mode="r") as infile:  # recover sequences
            sequences = [line.strip() for line in infile if not line.startswith(">")]
    except OSError as e:
        raise OSError(f"An error occurred while reading {fasta}") from e
    return {seqname: sequences[i] for i, seqname in enumerate(seqnames)}


def select_shuffle_bg_size(fastapos: str, fastaneg: str, prefix: str) -> None:
    """ """
    # retrieve sequence names for positive and negative sequences
    sequencespos, sequencesneg = read_fasta(fastapos), read_fasta(fastaneg)
    selected_sequences = [f"{seqname}_shuf" for seqname in sequencespos]
    # write negative train sequences fasta
    fastabg_size = f"{prefix}_shuffle_neg_train.fa"
    try:
        with open(fastabg_size, mode="w") as outfile:
            for seqname in selected_sequences:
                outfile.write(f">{seqname}\n{sequencesneg[seqname]}\n")
    except OSError as e:
        raise OSError("An error occurred while writing shuffle background sequences") from e
    

def chrom_split(fname: str) -> Dict[str, List[str]]:
    """ """
    chrom_dict = {c: [] for c in CHROMS}
    with open(fname, mode="r") as infile:
        for line in infile:  # group each feature in bed by chromosome
            chrom = line.strip().split()[0]
            chrom_dict[chrom].append(line)
    return chrom_dict

def compute_dnase_bg_size(trainpos: str, trainneg: str, prefix: str) -> Tuple[str, str]:
    """ """
    # split genomic features by chromosome
    trainpos_chrom, trainneg_chrom = chrom_split(trainpos), chrom_split(trainneg)
    bgtrain = []
    for chrom in trainpos_chrom:  # select train features
        bgtrain.extend(random.sample(trainneg_chrom[chrom], len(trainpos_chrom[chrom])))
    trainneg_fname = f"{prefix}_dnase_neg_train.bed"
    with open(trainneg_fname, mode="w") as outfile:
        outfile.write("".join(bgtrain))
    return trainneg_fname


def compute_datasets_size(benchdir: str, traindatadir: str, genome: str, shuffle: bool):
    """ """
    # define comparison root directory
    rootdir = os.path.join(benchdir, "dataset-size-comparison")
    trainposdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/positive") if shuffle else os.path.join(rootdir, "dnase/train/fasta/positive")
    trainposdir_bed = os.path.join(rootdir, "shuffle/train/bed/positive") if shuffle else os.path.join(rootdir, "dnase/train/bed/positive")
    trainnegdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/negative") if shuffle else os.path.join(rootdir, "dnase/train/fasta/negative")
    trainnegdir_bed = os.path.join(rootdir, "shuffle/train/bed/negative") if shuffle else os.path.join(rootdir, "dnase/train/bed/negative")
    traindatadir = os.path.join(traindatadir, "shuffle") if shuffle else os.path.join(traindatadir, "dnase")
    # retrieve positive bed files
    posbeds = glob(os.path.join(traindatadir, "positive/*.bed"))
    for size in SIZES:  # iterate over dataset sizes
        sizedir = "size_full" if size == 0 else f"size_{size}"
        sys.stdout.write(f"size - {sizedir}\n")
        trainposdir_fasta_size = os.path.join(trainposdir_fasta, sizedir)
        trainposdir_bed_size = os.path.join(trainposdir_bed, sizedir)
        trainnegdir_fasta_size = os.path.join(trainnegdir_fasta, sizedir)
        trainnegdir_bed_size = os.path.join(trainnegdir_bed, sizedir)
        for d in [trainposdir_fasta_size, trainposdir_bed_size, trainnegdir_fasta_size, trainnegdir_bed_size]:
            if not os.path.isdir(d):  # create outout folders if not already present
                os.makedirs(d)
        for posbed in tqdm(posbeds):  # iterate over positive bed files
            bed_prefix = os.path.basename(posbed).split("_")[0]
            bed_pos = sort_bed(read_bed(posbed), os.path.join(trainposdir_bed_size, bed_prefix), size)
            fastapos = extract_sequences(bed_pos, genome)
            if shuffle:  # compute synthetic background data
                fastaneg = os.path.join(traindatadir, "negative", f"{bed_prefix}_shuffle_neg_train.fa")
                # select background sequences based on the picked positive sequences
                select_shuffle_bg_size(fastapos, fastaneg, os.path.join(trainnegdir_fasta_size, bed_prefix))
            else:  # compute real biological background data
                negbed = os.path.join(traindatadir, "negative", f"{bed_prefix}_dnase_neg_train.bed")
                bed_neg = compute_dnase_bg_size(bed_pos.fn, negbed, os.path.join(trainnegdir_bed_size, bed_prefix))
                fastaneg = extract_sequences(pybedtools.BedTool(bed_neg), genome)
                subprocess.call(f"mv {fastaneg} {trainnegdir_fasta_size}", shell=True)
            subprocess.call(f"mv {fastapos} {trainposdir_fasta_size}", shell=True)


def resize_peak(bedline: List[str], width: int) -> str:
    """ """
    # retrieve start, stop and summit position
    start, stop, summit = list(map(int, bedline[1:3] + [bedline[9]]))
    center = start + summit  # compute peak summit position
    # compute new start and stop positions centered around peak summit
    start = str(center - int(width / 2)) if width != 0 else str(start)
    stop = str(center + int(width / 2)) if width != 0 else str(stop)
    return "\t".join(bedline[:1] + [start, stop] + bedline[3:])


def resize_bed_peaks(bedfile: str, width: int, prefix: str) -> pybedtools.BedTool:
    """ """
    try:  # read input peaks 
        with open(bedfile, mode="r") as infile:
            bedlines = [line.strip().split() for line in infile]
    except OSError as e:
        raise OSError(f"An error occurred while reading {bedfile}") from e
     # resize peaks according input width
    bedlines_resized = [resize_peak(line, width) for line in bedlines] 
    bedout = f"{prefix}_train.bed"
    try:
        with open(bedout, mode="w") as outfile:  # write resizied peaks 
            outfile.write("\n".join(bedlines_resized))
    except OSError as e:
        raise OSError(f"An error occurred while resizing peaks on {bedfile}") from e
    return pybedtools.BedTool(bedout)


def compute_datasets_width(benchdir: str, traindatadir: str, genome: str, shuffle: bool):
    """ """
    # define comparison root directory
    rootdir = os.path.join(benchdir, "sequence-width-comparison")
    trainposdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/positive") if shuffle else os.path.join(rootdir, "dnase/train/fasta/positive")
    trainposdir_bed = os.path.join(rootdir, "shuffle/train/bed/positive") if shuffle else os.path.join(rootdir, "dnase/train/bed/positive")
    trainnegdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/negative") if shuffle else os.path.join(rootdir, "dnase/train/fasta/negative")
    trainnegdir_bed = os.path.join(rootdir, "shuffle/train/bed/negative") if shuffle else os.path.join(rootdir, "dnase/train/bed/negative")
    traindatadir = os.path.join(traindatadir, "shuffle") if shuffle else os.path.join(traindatadir, "dnase")
    # retrieve positive bed files
    posbeds = glob(os.path.join(traindatadir, "positive/*.bed"))
    for width in WIDTHS:  # iterate over dataset sizes
        widthdir = "width_full" if width == 0 else f"width_{width}"
        sys.stdout.write(f"width - {widthdir}\n")
        trainposdir_fasta_width = os.path.join(trainposdir_fasta, widthdir)
        trainposdir_bed_width = os.path.join(trainposdir_bed, widthdir)
        trainnegdir_fasta_width = os.path.join(trainnegdir_fasta, widthdir)
        trainnegdir_bed_width = os.path.join(trainnegdir_bed, widthdir)
        for d in [trainposdir_fasta_width, trainposdir_bed_width, trainnegdir_fasta_width, trainnegdir_bed_width]:
            if not os.path.isdir(d):  # create outout folders if not already present
                os.makedirs(d)
        for posbed in tqdm(posbeds):  # iterate over positive bed files
            bed_prefix = os.path.basename(posbed).split("_")[0]
            bed_pos = resize_bed_peaks(posbed, width, os.path.join(trainposdir_bed_width, bed_prefix))
            fastapos = extract_sequences(bed_pos, genome)
            if shuffle:  # compute synthetic background data
                fastaneg = os.path.join(traindatadir, "negative", f"{bed_prefix}_shuffle_neg_train.fa")
                subprocess.call(f"cp {fastaneg} {trainnegdir_fasta_width}", shell=True)
            else:  # compute real biological background data
                bedneg = os.path.join(traindatadir, "negative", f"{bed_prefix}_dnase_neg_train.bed")
                subprocess.call(f"cp {bedneg} {trainnegdir_bed_width}", shell=True)
                fastaneg = os.path.join(traindatadir, "negative", f"{bed_prefix}_dnase_neg_train.fa")
                subprocess.call(f"cp {fastaneg} {trainnegdir_fasta_width}", shell=True)
            subprocess.call(f"mv {fastapos} {trainposdir_fasta_width}", shell=True)

def read_perf_table(fname: str) -> pd.DataFrame:
    return pd.read_csv(fname, sep="\t")

def retrieve_best_performance_global(benchdir: str, tool: str) -> Tuple[int, int]:
    # compute best performing dataset size
    sizes_perf = {size: 0 for size in SIZES}
    for size in SIZES:
        size_name = "size_full" if size == 0 else f"size_{size}"
        table_fname = f"summary_table_{size_name}_{tool}.tsv"
        perftable = read_perf_table(os.path.join(benchdir, "performance/dataset-size-comparison/dnase-dnase", table_fname))
        sizes_perf[size] = np.mean(perftable["AUPRC"])
    # compute best performing sequence width
    widths_perf = {width: 0 for width in WIDTHS}
    for width in WIDTHS:
        width_name = "width_full" if width == 0 else f"width_{width}"
        table_fname = f"summary_table_{width_name}_{tool}.tsv"
        perftable = read_perf_table(os.path.join(benchdir, "performance/sequence-width-comparison/dnase-dnase", table_fname))
        widths_perf[width] = np.mean(perftable["AUPRC"])
    return max(sizes_perf, key=sizes_perf.get), max(widths_perf, key=widths_perf.get)


def compute_datasets_optimal_global(benchdir: str, compdatadir: str, traindatadir: str, genome: str, shuffle: bool):
    # define comparison root directory
    rootdir = os.path.join(compdatadir, "optimal-global")
    trainposdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/positive") if shuffle else os.path.join(rootdir, "dnase/train/fasta/positive")
    trainposdir_bed = os.path.join(rootdir, "shuffle/train/bed/positive") if shuffle else os.path.join(rootdir, "dnase/train/bed/positive")
    trainnegdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/negative") if shuffle else os.path.join(rootdir, "dnase/train/fasta/negative")
    trainnegdir_bed = os.path.join(rootdir, "shuffle/train/bed/negative") if shuffle else os.path.join(rootdir, "dnase/train/bed/negative")
    traindatadir = os.path.join(traindatadir, "shuffle") if shuffle else os.path.join(traindatadir, "dnase")
    # retrieve positive bed files
    posbeds = glob(os.path.join(traindatadir, "positive/*.bed"))
    for tool in TOOLS[:2]:  
        # recover the best performing (globally) sequence width and dataset size 
        size, width = retrieve_best_performance_global(benchdir, tool)
        for d in [trainposdir_fasta, trainposdir_bed, trainnegdir_fasta, trainnegdir_bed]:
            os.makedirs(d, exist_ok=True)  # create outout folders if not already present
        for posbed in tqdm(posbeds):
            bed_prefix = os.path.basename(posbed).split("_")[0] + f"_{tool}"
            bedsize = sort_bed(read_bed(posbed), os.path.join(trainposdir_bed, bed_prefix), size)
            bedwidth = resize_bed_peaks(bedsize.fn, width, os.path.join(trainposdir_bed, bed_prefix))
            fastapos = extract_sequences(bedwidth, genome)
            bed_prefix_ = bed_prefix.replace(f"_{tool}", "")
            size_name = "size_full" if size == 0 else f"size_{size}"
            compsizedir = os.path.join(compdatadir, "dataset-size-comparison")
            if shuffle:  # compute synthetic background data
                fastaneg = os.path.join(compsizedir, f"shuffle/train/fasta/negative/{size_name}", f"{bed_prefix_}_shuffle_neg_train.fa")
                fastaneg_target = os.path.join(trainnegdir_fasta, f"{bed_prefix}_shuffle_neg_train.fa")
                subprocess.call(f"cp {fastaneg} {fastaneg_target}", shell=True)
            else:  # compute real biological background data
                bedneg = os.path.join(compsizedir, f"dnase/train/bed/negative/{size_name}", f"{bed_prefix_}_dnase_neg_train.bed")
                bedneg_target = os.path.join(trainnegdir_bed, f"{bed_prefix}_dnase_neg_train.bed")
                subprocess.call(f"cp {bedneg} {bedneg_target}", shell=True)
                fastaneg = os.path.join(compsizedir, f"dnase/train/fasta/negative/{size_name}", f"{bed_prefix_}_dnase_neg_train.fa")
                fastaneg_target = os.path.join(trainnegdir_fasta, f"{bed_prefix}_dnase_neg_train.fa")
                subprocess.call(f"cp {fastaneg} {fastaneg_target}", shell=True)
            subprocess.call(f"mv {fastapos} {trainposdir_fasta}", shell=True)


def retrieve_best_performance_local(benchdir: str, experiment_name: str, tool: str) -> Tuple[int, int]:
    # compute best performing dataset size
    sizes_perf = {size: 0 for size in SIZES}
    for size in SIZES:
        size_name = "size_full" if size == 0 else f"size_{size}"
        table_fname = f"summary_table_{size_name}_{tool}.tsv"
        perftable = read_perf_table(os.path.join(benchdir, "performance/dataset-size-comparison/dnase-dnase", table_fname))
        perftable.set_index("EXPERIMENT", inplace=True)
        sizes_perf[size] = perftable.loc[experiment_name, "AUPRC"]
    # compute best performing sequence width
    widths_perf = {width: 0 for width in WIDTHS}
    for width in WIDTHS:
        width_name = "width_full" if width == 0 else f"width_{width}"
        table_fname = f"summary_table_{width_name}_{tool}.tsv"
        perftable = read_perf_table(os.path.join(benchdir, "performance/sequence-width-comparison/dnase-dnase", table_fname))
        perftable.set_index("EXPERIMENT", inplace=True)
        widths_perf[width] = perftable.loc[experiment_name, "AUPRC"]
    return max(sizes_perf, key=sizes_perf.get), max(widths_perf, key=widths_perf.get)


def compute_datasets_optimal_local(benchdir: str, compdatadir: str, traindatadir: str, genome: str, shuffle: bool):
    # define comparison root directory
    rootdir = os.path.join(compdatadir, "optimal-local")
    trainposdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/positive") if shuffle else os.path.join(rootdir, "dnase/train/fasta/positive")
    trainposdir_bed = os.path.join(rootdir, "shuffle/train/bed/positive") if shuffle else os.path.join(rootdir, "dnase/train/bed/positive")
    trainnegdir_fasta = os.path.join(rootdir, "shuffle/train/fasta/negative") if shuffle else os.path.join(rootdir, "dnase/train/fasta/negative")
    trainnegdir_bed = os.path.join(rootdir, "shuffle/train/bed/negative") if shuffle else os.path.join(rootdir, "dnase/train/bed/negative")
    traindatadir = os.path.join(traindatadir, "shuffle") if shuffle else os.path.join(traindatadir, "dnase")
    # retrieve positive bed files
    posbeds = glob(os.path.join(traindatadir, "positive/*.bed"))
    for tool in TOOLS[:2]:  
        for d in [trainposdir_fasta, trainposdir_bed, trainnegdir_fasta, trainnegdir_bed]:
            os.makedirs(d, exist_ok=True)  # create outout folders if not already present
        for posbed in tqdm(posbeds):
            experiment_name = os.path.basename(posbed).split("_")[0] 
            size, width = retrieve_best_performance_local(benchdir, experiment_name, tool)
            bed_prefix = f"{experiment_name}_{tool}"
            bedsize = sort_bed(read_bed(posbed), os.path.join(trainposdir_bed, bed_prefix), size)
            bedwidth = resize_bed_peaks(bedsize.fn, width, os.path.join(trainposdir_bed, bed_prefix))
            fastapos = extract_sequences(bedwidth, genome)
            bed_prefix_ = bed_prefix.replace(f"_{tool}", "")
            size_name = "size_full" if size == 0 else f"size_{size}"
            compsizedir = os.path.join(compdatadir, "dataset-size-comparison")
            if shuffle:  # compute synthetic background data
                fastaneg = os.path.join(compsizedir, f"shuffle/train/fasta/negative/{size_name}", f"{bed_prefix_}_shuffle_neg_train.fa")
                fastaneg_target = os.path.join(trainnegdir_fasta, f"{bed_prefix}_shuffle_neg_train.fa")
                subprocess.call(f"cp {fastaneg} {fastaneg_target}", shell=True)
            else:  # compute real biological background data
                bedneg = os.path.join(compsizedir, f"dnase/train/bed/negative/{size_name}", f"{bed_prefix_}_dnase_neg_train.bed")
                bedneg_target = os.path.join(trainnegdir_bed, f"{bed_prefix}_dnase_neg_train.bed")
                subprocess.call(f"cp {bedneg} {bedneg_target}", shell=True)
                fastaneg = os.path.join(compsizedir, f"dnase/train/fasta/negative/{size_name}", f"{bed_prefix_}_dnase_neg_train.fa")
                fastaneg_target = os.path.join(trainnegdir_fasta, f"{bed_prefix}_dnase_neg_train.fa")
                subprocess.call(f"cp {fastaneg} {fastaneg_target}", shell=True)
            subprocess.call(f"mv {fastapos} {trainposdir_fasta}", shell=True)


def main():
    # parse input arguments (comparison to perform and benchmark data base folder)
    # the script assumes that data directory follows the structure defined in the 
    # previous step of the pipeline
    comparison, traindatadir, genome, benchdir = parse_commandline(sys.argv[1:])
    if not benchdir:
        os.mkdir(benchdir)  # if not aslready present, create benchmark data folder
    if comparison == COMPARISONS[0]:  # benchmark performance on dataset size
        compute_datasets_size(benchdir, traindatadir, genome, True)  # synthetic bg
        compute_datasets_size(benchdir, traindatadir, genome, False)  # real bg
    elif comparison == COMPARISONS[1]:  # benchmark perfomance on sequences width
        compute_datasets_width(benchdir, traindatadir, genome, True)  # synthetic bg
        compute_datasets_width(benchdir, traindatadir, genome, False)  # real bg
    elif comparison == COMPARISONS[2]:  # benchmark performance on optimal global features
        compute_datasets_optimal_global(benchdir, "comparison-data", traindatadir, genome, True)  # synthetic bg
        compute_datasets_optimal_global(benchdir, "comparison-data", traindatadir, genome, False)  # real bg
    elif comparison == COMPARISONS[3]: # benchmark performance on optimal local features
        compute_datasets_optimal_local(benchdir, "comparison-data", traindatadir, genome, True)  # synthetic bg
        compute_datasets_optimal_local(benchdir, "comparison-data", traindatadir, genome, False)  # real bg


if __name__ == "__main__":
    main()



