#!/bin/bash
# To run: ./scripts/slurm/submit_index_subsets.sh "ms2" "/work/frink/li.mil/data/marginalization-datasets"

# SUBSETS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
SUBSETS=(10 15 20)
# SUBSETS=(1)
DATASET=$1
OUTPUT_DIR=$2

if [[ "$DATASET" == "ms2" ]]; then
    # Use the 1000 relevance cutoff for MS2
    for i in "${SUBSETS[@]}";
    do
        sbatch "./scripts/slurm/index_subsets.sh" $DATASET \
            "${OUTPUT_DIR}/${DATASET}_retrieved_split=${i}" \
            $i \
            1000 \
            "/scratch/li.mil/open-domain-mds-merge/contriever_msmarco_ms2/model_best" \
            "/scratch/li.mil/open-domain-mds-merge/platt_calibration/ms2/logreg_1000.pkl"
    done
elif [[ "$DATASET" == "cochrane" ]]; then
    # Use the 500 relevance cutoff for Cochrane
    for i in "${SUBSETS[@]}"
    do
        sbatch "./scripts/slurm/index_subsets.sh" $DATASET \
            "${OUTPUT_DIR}/${DATASET}_retrieved_split=${i}" \
            $i \
            500 \
            "/scratch/li.mil/open-domain-mds-merge/contriever_cochrane/model_best" \
            "/scratch/li.mil/open-domain-mds-merge/platt_calibration/cochrane/logreg_500.pkl"
    done
else
    printf '%s\n' "Pick a valid dataset!" >&2
    exit 1
fi