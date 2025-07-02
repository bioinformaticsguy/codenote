#!/bin/bash

#SBATCH --partition=shortterm
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=10GB
#SBATCH --time=0-1:00:00
#SBATCH --tmp=100G
#SBATCH --job-name=Count_Motif_Biopython
#SBATCH --output=motif_count_biopython.out
#SBATCH --error=motif_count_biopython.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sudharsan.chandra@uni-luebeck.de

# Activate your virtual environment
source ~/mpra_env/bin/activate

# Run the Python script
python3 <<EOF
from Bio import SeqIO

motif = "CCGGGCTCGAGATCTGCGATCTAAGTAAGC"
motif_rc = motif.translate(str.maketrans("ACGT", "TGCA"))[::-1]

input_file = "AR_library.fastq"
match_count = 0
total = 0

with open(input_file, "r") as handle:
    for record in SeqIO.parse(handle, "fastq"):
        seq = str(record.seq)
        if motif in seq or motif_rc in seq:
            match_count += 1
        total += 1

print(f"Total reads: {total}")
print(f"Reads with motif (fwd or rev): {match_count}")
EOF
