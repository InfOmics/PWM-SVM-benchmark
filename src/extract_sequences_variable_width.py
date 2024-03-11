import pandas as pd
import os
import sys
import subprocess

def extract_sequences(input_dir: str, output_dir: str, length: int, ref_genome: str) -> None:
    for folder in os.listdir(input_dir):
        for file in os.listdir(os.path.join(input_dir, folder)):
            bedfile_path = os.path.join(input_dir, folder, file)
            bedfile = pd.read_csv(bedfile_path, sep = "\t", header = None)
            df_seqs = pd.DataFrame(columns = bedfile.columns)
            for i,r in bedfile.iterrows():
                row = r.copy() # to not overwrite r
                summit = row[9]
                if length != 0:
                    row[1] = r[1] + summit - int(length/2) - 1 # the center of the sequence is the summit
                    row[2] = r[1] + summit + int(length/2)
                else:
                    row[1] = r[1]
                    row[2] = r[2]
                df_seqs = pd.concat([df_seqs, row.to_frame().T], ignore_index=True)
            os.makedirs(os.path.join(output_dir, f"Width_{length}" if length != 0 else "Full_width", os.path.splitext(os.path.basename(bedfile_path))[0]), exist_ok=True)
            filename = f"{os.path.splitext(os.path.basename(bedfile_path))[0]}_{f'Width{length}' if length != 0 else 'full_length'}.bed"
            outfile = os.path.join(output_dir, f"Width_{length}" if length != 0 else "Full_width", os.path.splitext(os.path.basename(bedfile_path))[0],filename)
            df_seqs.to_csv(outfile, sep = "\t", header = False, index = False)
            print(f"Creato file {outfile}")
            
            #get fasta from bed file
            output_name = f"{os.path.splitext(os.path.basename(bedfile_path))[0]}_{f'Width{length}' if length != 0 else 'full_length'}.fa"
            outpath = os.path.join(output_dir, f"Width_{length}" if length != 0 else "Full_width", os.path.splitext(os.path.basename(bedfile_path))[0],output_name)

            subprocess.call(f"bedtools getfasta -fi {ref_genome} -bed {bedfile_path} -fo {outpath}", shell=True)
            print(f"Creato file {output_name}")


def main():
    input_dir ,output_dir, ref_genome = sys.argv[1:]
    os.makedirs(output_dir, exist_ok=True)
    LENGHTS = [50, 100, 150, 200, 0]
    for length in LENGHTS:
        extract_sequences(input_dir,output_dir, length, ref_genome)
        
if __name__ == "__main__":
    main()