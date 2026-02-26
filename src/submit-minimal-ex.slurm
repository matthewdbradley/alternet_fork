#!/bin/bash

#SBATCH -N 1 

#SBATCH -c 16

#SBATCH --mem-per-cpu=4G 

#SBATCH --time=0-12:00:00  

#SBATCH --output=alternet.output

#SBATCH --mail-user=mxb1123@case.edu 

#SBATCH --mail-type=END

#SBATCH --job-name="alternet" 

#conda activate alternet
cd /home/mxb1123/alternet_fork/src
python minimal-working-ex.py 
