#!/bin/bash

JOB_NAME="Single_Galaxies_CNN_Train"
LOG_DIR="/projects/bdsp/wenyinli/models/LOGS/$JOB_NAME"
RUN_NAME="$JOB_NAME"
OUTPUT_DIR_RUN="/projects/bdsp/wenyinli/models/$RUN_NAME/"

cat <<EOF > job_script.sbatch
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpuH200x8
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdsp-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 1:00:00
#SBATCH --output=$LOG_DIR/output.%j.%N.out
#SBATCH --error=$LOG_DIR/error.%j.%N.out

conda init
conda activate /projects/bdsp/miniconda3/envs/dd_shurui

python ./single_gal_CNN.py
EOF

sbatch job_script.sbatch
