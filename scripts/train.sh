#!/bin/bash

JOB_NAME="D4T_CNN"
LOG_DIR="/projects/bdsp/wenyinli/models/LOGS/$JOB_NAME"
RUN_NAME="$JOB_NAME"
OUTPUT_DIR_RUN="/projects/bdsp/wenyinli/models/$RUN_NAME/"
STORAGE="sqlite:////projects/bdsp/wenyinli/models/optuna_studies/$JOB_NAME.db"

cat <<EOF > job_script.sbatch
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpuA100x4
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bdsp-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 4:00:00
#SBATCH --output=$LOG_DIR/output.%j.%N.out
#SBATCH --error=$LOG_DIR/error.%j.%N.out

conda init
conda activate /u/wenyinli/.conda/envs/anacal_env

# Each worker can run some number of trials
PYTHONPATH=/projects/bdsp/wenyinli/codes/single_galaxies

NGPU=4
TRIALS_PER_GPU=25
for GPU in \$(seq 0 \$((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=\$GPU \\
    python -u ./src/training.py \\
        --images /projects/bdsp/wenyinli/datasets/simple_gal_images.npy \\
        --csv /projects/bdsp/wenyinli/datasets/simple_gal_info.csv \\
        --target e \\
        --epochs 200 \\
        --num-workers 8 \\
        --study-name $JOB_NAME \\
        --storage $STORAGE \\
        --n-trials \$TRIALS_PER_GPU \\
        --device cuda \\
        --seed \$((520 + GPU)) \\
        > $LOG_DIR/gpu\${GPU}.out 2>&1 &

done

wait
echo "All GPU workers finished."
EOF

sbatch job_script.sbatch