#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=finetunemocounet
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=127000M
#SBATCH --time=12:00:00

#export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
#export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

module load python/3.11
source /home/codee/miniconda3/etc/profile.d/conda.sh
conda activate base
#conda conda install -c conda-forge opencv

#config="/home/codee/scratch/sourcecode/cem-dataset/pretraining/mocov2/mocov2_config.yaml"

log_dir="/home/codee/scratch/sourcecode/cem-dataset/evaluation/finetunesave"
echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

#echo "$SLURM_NODEID master: $MASTER_ADDR"
echo "$SLURM_NODEID Launching python script"
#echo "$SLURM_NTASKS tasks running"

/home/codee/miniconda3/bin/python /home/codee/scratch/sourcecode/cem-dataset/evaluation/finetune.py > $log_dir/outputfinuetune

#python /home/codee/scratch/sourcecode/cem-dataset/evaluation/setup_benchmarks/setup_data.py "/home/codee/scratch/sourcecode/cem-dataset/benchdata"

echo "finetune finished"