""" 
"""

import subprocess
import sys
import os

MEMEDEFAULT = "-dna -mod zoops -nmotifs 1 -minw 6 -maxw 30 -revcomp"

def gkmtrain(positive: str, negative: str, outdir: str) -> str:
    modelprefix = os.path.join(outdir, os.path.splitext(os.path.basename(positive))[0])
    try:
        subprocess.run(["gkmtrain", "-T", "16", positive, negative, modelprefix])
    except OSError as e:
        raise OSError("SVM training failed")
    return f"{modelprefix}.model.txt"

def meme(positive: str, outdir: str) -> str:
    outdir = os.path.join(outdir, os.path.splitext(os.path.basename(positive))[0])
    try:
        subprocess.call(f"meme -oc {outdir} {MEMEDEFAULT} {positive}", shell=True)
    except OSError as e:
        raise OSError("MEME training failed")
    return os.path.join(outdir, "meme.txt")


def main() -> None:
    positive, negative, outdir = sys.argv[1:]
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # train SVM model
    gkmtrain(positive, negative, outdir)
    # train PWM model
    meme(positive, outdir)

if __name__ == "__main__":
    main()





