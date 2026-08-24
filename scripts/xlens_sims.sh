#!/bin/bash
JOB_NAME="xlens_sims"
OUTPUT_DIR="/projects/bdsp/wenyinli/datasets/$JOB_NAME"
RUN_NAME="$JOB_NAME"
OUTPUT_DIR_RUN="/work/hdd/bdsp/wenyinli/datasets/$JOB_NAME"

cat <<EOF > job_script.sbatch
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
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
#SBATCH --output=$OUTPUT_DIR/%x.%j.out
#SBATCH --error=$OUTPUT_DIR/%x.%j.err

source /projects/bdsp/miniconda3/etc/profile.d/conda.sh
conda activate /projects/bdsp/miniconda3/envs/lsst-scipipe-10.1.0
setup lsst_distrib

python src/datasets/xlens_gal_sim.py 


EOF

# Submit the job
sbatch job_script.sbatch
