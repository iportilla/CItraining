#!/bin/bash
#SBATCH -J model-tuning-1H
#SBATCH -o %x.out%j
#SBATCH -p main
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1

module load container_env
module load tensorflow-gpu/2.6.0
CRUN=crun.tensorflow-gpu
CRUN_ENVS_FLAGS="-p $HOME/envs/default-tensorflow-gpu-2.6.0"

# HYPERPARAMS_LIST contains four arguments,
# which must be given in the command-line argument:
#   * the number of hidden neurons
#   * the learning rate
#   * the batch size
#   * the number of epochs to train

HIDDEN_NEURONS=$1
LEARNING_RATE=$2
BATCH_SIZE=$3
EPOCHS=$4

##BEGIN_DEVELOPER_CODE
export FORCE_REPRODUCIBLE_TRAINING=yes

# Print key diagnostics information -- external environment

echo "#BEGIN ENVIRONMENT DIAGNOSTICS INFORMATION ---"
echo "  date:      $(date)"
echo "  hostname:  $(hostname)"
echo "  user:      $(whoami)"
echo "  container: ${CRUN}"
echo "  container_fullpath: $(which ${CRUN})"
echo "  python3:   $($CRUN $CRUN_ENVS_FLAGS which python3)"
#echo "  python3_realpath:   $($CRUN $CRUN_ENVS_FLAGS which python3)"
echo "  python3_version::"
$CRUN $CRUN_ENVS_FLAGS  python3 --version
echo "#END ENVIRONMENT DIAGNOSTICS INFORMATION ---"

echo "SLURM DIAGNOSTICS:"
export | grep SLURM_
echo
##END_DEVELOPER_CODE


$CRUN $CRUN_ENVS_FLAGS  python3  NN_Model_1H_devel.py  "$HIDDEN_NEURONS"  "$LEARNING_RATE"  "$BATCH_SIZE"  "$EPOCHS"