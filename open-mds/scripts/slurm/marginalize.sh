#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=1-30%2
#SBATCH --job-name=marginalize
#SBATCH --partition=177huntington
#SBATCH --output=/home/li.mil/open-domain-mds-merge/open-mds/logs/marginalize_%j.out
#SBATCH --error=/home/li.mil/open-domain-mds-merge/open-mds/logs/marginalize_%j.err

eval "$(conda shell.bash hook)"
conda activate mds
source ~/.bashrc

DATASET=$1
OUTPUT_DIR=$2
MODEL=$3

if [[ "${MODEL}" =~ "vanilla" ]]; then
    python ./scripts/uncertainty_run_summarization.py "./conf_updated/base.yml" "./conf_updated/${DATASET}/led-base/eval.yml" \
        output_dir="${OUTPUT_DIR}/${DATASET}_retrieved_split=$SLURM_ARRAY_TASK_ID/led-base/vanilla" \
        dataset_name="/work/frink/li.mil/data/marginalization-datasets/${DATASET}_retrieved_split=$SLURM_ARRAY_TASK_ID/" \
        model_name_or_path="${MODEL}"

else
    python ./scripts/uncertainty_run_summarization.py "./conf_updated/base.yml" "./conf_updated/${DATASET}/led-base/eval.yml" \
        output_dir="${OUTPUT_DIR}/${DATASET}_retrieved_split=$SLURM_ARRAY_TASK_ID/led-base/reretrieved" \
        dataset_name="/work/frink/li.mil/data/marginalization-datasets/${DATASET}_retrieved_split=$SLURM_ARRAY_TASK_ID/" \
        model_name_or_path="${MODEL}"
fi