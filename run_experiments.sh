#!/bin/bash

# This script runs the ns_run.py script with all 8 combinations of
# --share/--no-share, --refinement/--no-refinement, and --picard/--no-picard.

set -e # Exit immediately if a command exits with a non-zero status.

EPOCHS=10000

# Create a top-level directory for all runs
mkdir -p runs

gpu_id=0
for share in "true" "false"; do
  for refinement in "true" "false"; do
    for picard in "true" "false"; do

      SHARE_FLAG="--share"
      SHARE_NAME_PART="sT"
      if [ "$share" = "false" ]; then
        SHARE_FLAG="--no-share"
        SHARE_NAME_PART="sF"
      fi

      REFINEMENT_FLAG="--refinement"
      REFINEMENT_NAME_PART="rT"
      TRAIN_KIND="acausal"
      VAL_KIND="acausal"
      if [ "$refinement" = "false" ]; then
        REFINEMENT_FLAG="--no-refinement"
        REFINEMENT_NAME_PART="rF"
        TRAIN_KIND="one_step"
        VAL_KIND="generate"
      fi

      PICARD_FLAG="--picard"
      PICARD_NAME_PART="pT"
      if [ "$picard" = "false" ]; then
        PICARD_FLAG="--no-picard"
        PICARD_NAME_PART="pF"
      fi

      RUN_NAME="${SHARE_NAME_PART}_${REFINEMENT_NAME_PART}_${PICARD_NAME_PART}"

      echo "====================================================================="
      echo "Running with: share=$share, refinement=$refinement, picard=$picard on GPU $gpu_id"
      echo "Run results will be in: $RUN_NAME"
      echo "====================================================================="

      nohup bash -c "CUDA_VISIBLE_DEVICES=$gpu_id exec -a \"$RUN_NAME\" python ns_run.py \
        --epochs $EPOCHS \
        --name \"$RUN_NAME\" \
        $SHARE_FLAG \
        $PICARD_FLAG \
        --train-kind $TRAIN_KIND \
        --val-kind $VAL_KIND" > "runs/$RUN_NAME.log" 2>&1 &

      gpu_id=$((gpu_id + 1))
    done
  done
done

wait

echo "All 8 runs completed successfully."