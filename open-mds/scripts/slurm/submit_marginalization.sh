#!/bin/bash
# To run: ./scripts/slurm/submit_marginalization.sh "ms2" /scratch/li.mil/open-domain-mds-merge/output allenai/led-base-16384-cochrane
# /scratch/li.mil/open-domain-mds-merge/models/cochrane/led-base/vanilla/checkpoint-4000
# /scratch/li.mil/open-domain-mds-merge/models/cochrane/led-base/retrieved_new/checkpoint-7500

STRATEGIES=("max" "mean" "oracle")

DATASET=$1
OUTPUT_DIR=$2
MODEL=$3

if [[ "$DATASET" == "ms2" ]] || [[ "$DATASET" == "cochrane" ]]; then
    # Submit the vanilla evaluation over the open domain mds work
    for strategy in "${STRATEGIES[@]}";
    do
        sbatch "./scripts/slurm/run_vanilla_evaluation.sh" $DATASET \
            "${OUTPUT_DIR}" \
            "${MODEL}" \
            "${strategy}"
    done

    # Submit the job array of all of the jobs we care about
    sbatch "./scripts/slurm/marginalize.sh" $DATASET \
        "${OUTPUT_DIR}" \
        "${MODEL}" \

else
    printf '%s\n' "Pick a valid dataset!" >&2
    exit 1
fi