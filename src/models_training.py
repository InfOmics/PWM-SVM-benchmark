""" 
"""

from typing import List, Tuple

import subprocess
import sys
import os

TOOLS = ["lsgkm", "meme", "streme"]
MEMEDEFAULT = "-dna -mod zoops -nmotifs 1 -minw 6 -maxw 30 -revcomp"
STREMEDEFAULT = "--objfun de --seed 42 --dna --nmotifs 1 --minw 8 --maxw 15"

def parse_commandline(args: List[str]) -> Tuple[str, str, str, str]:
    if len(args) != 4:
        raise ValueError("Too many/few input arguments")
    positive, negative, tool, outdir = args
    if not os.path.isfile(positive):
        raise FileNotFoundError(f"Cannot find {positive}")
    if not os.path.isfile(negative):
        raise FileNotFoundError(f"Cannot find {negative}")
    if tool not in TOOLS:
        raise ValueError(f"Forbidden tool argument ({tool})")
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f"Cannot find {outdir}")
    return positive, negative, tool, outdir

def gkmtrain(positive: str, negative: str, outdir: str) -> str:
    # compute svm model using lsgkm
    outprefix = os.path.splitext(os.path.basename(positive))[0].split("_", 1)
    outdir = os.path.join(outdir, f"{outprefix}_svm")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    modelprefix = os.path.join(outdir, outprefix)
    try:  # run lsgkm
        code = subprocess.call(f"gkmtrain -T 16 {positive} {negative} {modelprefix}", shell=True)
        if code != 0:
            raise subprocess.SubprocessError(f"SVM training failed on {positive}")
    except OSError as e:
        raise OSError("SVM training failed") from e

def meme(positive: str, outdir: str) -> str:
    outprefix = os.path.splitext(os.path.basename(positive))[0].split("_", 1)
    outdir = os.path.join(outdir, f"{outprefix}_meme")
    try:
        code = subprocess.call(f"meme -oc {outdir} {MEMEDEFAULT} {positive}", shell=True)
        if code != 0:
            raise subprocess.SubprocessError(f"MEME PWM training failed on {positive}")
    except OSError as e:
        raise OSError("MEME training failed") from e

def streme(positive: str, negative: str, outdir: str) -> str:
    outprefix = os.path.splitext(os.path.basename(positive))[0].split("_", 1)
    outdir = os.path.join(outdir, f"{outprefix}_streme")
    try:
        code = subprocess.call(f"streme -oc {outdir} {STREMEDEFAULT} --n {negative} --p {positive}", shell=True)
        if code != 0:
            raise subprocess.SubprocessError(f"STREME PWM training failed on {positive}")
    except OSError as e:
        raise OSError("STREME training failed") from e


def main() -> None:
    # read input arguments
    positive, negative, tool, outdir = parse_commandline(sys.argv[1:])
    # train models
    if tool == TOOLS[0]:  # svm model
        gkmtrain(positive, negative, outdir)
    elif tool == TOOLS[1]:  # meme pwm 
        meme(positive, outdir)
    elif tool == TOOLS[2]:  # streme pwm
        streme(positive, negative, outdir)

if __name__ == "__main__":
    main()





