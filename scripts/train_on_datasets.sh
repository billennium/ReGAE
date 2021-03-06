#!/bin/bash

MODEL_NAME=$1
MODEL_TRAIN_MODULE=$2
DATASET_PATH=$3
TARGET_CHECKPOINT_DIR=$4
GPU=${5:-0}

current_seed=${6:-0}

DATASET_NAMES=("0.pkl" "1.pkl" "2.pkl" "3.pkl" "4.pkl")
DATASET_NAMES_RAW=${7}

if [ ! -z "${DATASET_NAMES_RAW}" ] #check if not empty
then
    echo "Specified custom datasets"
    IFS=',' read -r -a DATASET_NAMES <<< "${DATASET_NAMES_RAW}"
fi


best_checkpoint_paths=()

echo "GPU ${GPU}"
echo "Specified datasets: ${DATASET_NAMES[@]}"

for dataset in "${DATASET_NAMES[@]}"; do
    echo "Starting experiment ${MODEL_NAME} on dataset ${dataset} with seed ${current_seed}"

    return_value=$(python3.9 -m ${MODEL_TRAIN_MODULE} \
      --gpus="${GPU}" \
      --pickled_dataset_path="${DATASET_PATH}/${dataset}" \
      --early_stopping=True \
      --es_patience=20 \
      --seed=${current_seed} \
      | tail -10 \
    )

    best_checkpoint=$(echo "$return_value" | grep "Best model path" | awk '{print $NF}')
    echo "Best checkpoint for dataset ${dataset}, seed ${current_seed}: ${best_checkpoint}"
    best_checkpoint_paths+=(${best_checkpoint})

    mkdir -p "${TARGET_CHECKPOINT_DIR}"
    target_ckpt_path="${TARGET_CHECKPOINT_DIR}/${dataset%.*}.ckpt"
    cp "${best_checkpoint}" "${target_ckpt_path}"

    checkpoints_dir="$(dirname "${best_checkpoint}")"
    logs_dir="$(dirname "${checkpoints_dir}")"
    hparams_file_path="${logs_dir}/hparams.yaml"
    target_hparams_path="${TARGET_CHECKPOINT_DIR}/${dataset%.*}_hparams.yaml"
    cp "${hparams_file_path}" "${target_hparams_path}"

    echo "Copied the checkpoint to ${target_ckpt_path}"

    ((current_seed++))
done

