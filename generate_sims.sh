#!/bin/bash
JOB_NAME="moffat"
OUTPUT_DIR="/projects/bdsp/wenyinli/datasets/single_gal"
RUN_NAME="$JOB_NAME"
OUTPUT_DIR_RUN="/work/hdd/bdsp/wenyinli/datasets/single_gal"

cat <<EOF > job_script.sbatch
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=cpu
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint="scratch"
#SBATCH --account=bdsp-delta-cpu
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 2:00:00
#SBATCH --output=$OUTPUT_DIR/%x.%j.out
#SBATCH --error=$OUTPUT_DIR/%x.%j.err

export OMP_NUM_THREADS=8
source /projects/bdsp/miniconda3/etc/profile.d/conda.sh
conda activate /projects/bdsp/miniconda3/envs/lsst-scipipe-10.0.0


python src/generate_sims.py \\
  --storage $OUTPUT_DIR_RUN/${RUN_NAME}_1p \\
  --shear 0.01 0.0 \\
  --psf_model Moffat \\
  --psf_para_range 2 3 0.6 0.8 \\
  --hlr_range 0.6 1.2 \\
  --flux_range 1000 50000 \\
  --e_max 0.7 \\
  --shift_std 0.0 \\
  --n_range 1 5 \\
  --noise_std_range 0 0 \\
  --N 10000 \\
  --image_size 64 \\
  --pixel_scale 0.2 \\
  --batch_size 1000 \\
  --num_workers 32 

python src/generate_sims.py \\
  --storage $OUTPUT_DIR_RUN/${RUN_NAME}_1m \\
  --shear -0.01 0.0 \\
  --psf_model Moffat \\
  --psf_para_range 2 3 0.6 0.8 \\
  --hlr_range 0.6 1.2 \\
  --flux_range 1000 50000 \\
  --e_max 0.7 \\
  --shift_std 0.0 \\
  --n_range 1 5 \\
  --noise_std_range 0 0 \\
  --N 10000 \\
  --image_size 64 \\
  --pixel_scale 0.2 \\
  --batch_size 1000 \\
  --num_workers 32 

python src/generate_sims.py \\
  --storage $OUTPUT_DIR_RUN/${RUN_NAME}_2p \\
  --shear 0.0 0.01 \\
  --psf_model Moffat \\
  --psf_para_range 2 3 0.6 0.8 \\
  --hlr_range 0.6 1.2 \\
  --flux_range 1000 50000 \\
  --e_max 0.7 \\
  --shift_std 0.0 \\
  --n_range 1 5 \\
  --noise_std_range 0 0 \\
  --N 10000 \\
  --image_size 64 \\
  --pixel_scale 0.2 \\
  --batch_size 1000 \\
  --num_workers 32 

python src/generate_sims.py \\
  --storage $OUTPUT_DIR_RUN/${RUN_NAME}_2m \\
  --shear 0.0 -0.01 \\
  --psf_model Moffat \\
  --psf_para_range 2 3 0.6 0.8 \\
  --hlr_range 0.6 1.2 \\
  --flux_range 1000 50000 \\
  --e_max 0.7 \\
  --shift_std 0.0 \\
  --n_range 1 5 \\
  --noise_std_range 0 0 \\
  --N 10000 \\
  --image_size 64 \\
  --pixel_scale 0.2 \\
  --batch_size 1000 \\
  --num_workers 32 

EOF

# Submit the job
sbatch job_script.sbatch
