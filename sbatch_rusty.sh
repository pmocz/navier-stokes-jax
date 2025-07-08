#!/usr/bin/bash
#SBATCH --job-name=navierstokes
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00-01:00

module purge
module load cuda python/3.11

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/navier-stokes-jax-venv/bin/activate

srun python navier-stokes-jax.py
