#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=index_subsets
#SBATCH --partition=177huntington,frink
#SBATCH --output=/home/li.mil/open-domain-mds-merge/open-mds/logs/index_subsets_%j.out
#SBATCH --error=/home/li.mil/open-domain-mds-merge/open-mds/logs/index_subsets_%j.err

eval "$(conda shell.bash hook)"
conda activate mds
source ~/.bashrc

DATASET=$1
OUTPUT_DIR=$2
NUM_SUBSETS=$3
RELEVANCE_CUTOFF=$4
MODEL=$5
CLASSIFIER=$6

python ./scripts/uncertainty_index_and_retrieve.py \
        "$DATASET" "$OUTPUT_DIR" \
        "$NUM_SUBSETS" \
        --relevance-cutoff "$RELEVANCE_CUTOFF" \
        --model-name-or-path  "$MODEL" \
        --classifier-file "$CLASSIFIER"