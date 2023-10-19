#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=retriever_training
#SBATCH --partition=177huntington,frink
#SBATCH --output=/home/li.mil/open-domain-mds-merge/open-mds/logs/retriever_training_%j.out
#SBATCH --error=/home/li.mil/open-domain-mds-merge/open-mds/logs/retriever_training_%j.err

eval "$(conda shell.bash hook)"
conda activate mds
source ~/.bashrc

python ./scripts/finetune_retriever.py \
        "./conf_retriever/base.yml" \
        "./conf_retriever/ms2/led-base/finetune.yml" \
        model_name_or_path="facebook/contriever" \
        output_dir="/scratch/li.mil/open-domain-mds-merge/contriever_ms2" \
        do_train=True \
        do_eval=True