#!/bin/bash -l  

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --job-name=URL_Analyzer
#SBATCH --output=Output_UA.txt
#SBATCH --constraint=AVX2

enable_lmod

module load python/3.6
module load numpy scipy
module load pandas
module load scikit-learn
module load ipython
module load cuda/9.1
module load tensorflow/1.10
module load keras/2.2
module use /scratch-lustre/DeapSECURE/lmod
module load DeapSECURE


# TODO for user: Create the url_analyzer_script.py script
# from the url_analyzer.py.
python url_analyzer_script.py

