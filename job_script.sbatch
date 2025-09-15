#!/bin/bash
#SBATCH --job-name=xlens_sims
#SBATCH --partition=cpu
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="scratch"
#SBATCH --account=bdsp-delta-cpu
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 2:00:00
#SBATCH --output=/projects/bdsp/wenyinli/datasets/xlens_sims/%x.%j.out
#SBATCH --error=/projects/bdsp/wenyinli/datasets/xlens_sims/%x.%j.err

source /projects/bdsp/miniconda3/etc/profile.d/conda.sh
conda activate /projects/bdsp/miniconda3/envs/lsst-scipipe-10.1.0
setup lsst_distrib

python src/datasets/xlens_gal_sim.py 


