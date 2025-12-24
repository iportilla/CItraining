#!/bin/bash
#SBATCH -J Baseline_Model
#SBATCH -o %x.out%j
#SBATCH -p main
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1

module load container_env
module load tensorflow-gpu/2.6.0
CRUN=crun.tensorflow-gpu
CRUN_ENVS_FLAGS="-p $HOME/envs/default-tensorflow-gpu-2.6.0"

$CRUN $CRUN_ENVS_FLAGS  python3  Baseline_Model.py  