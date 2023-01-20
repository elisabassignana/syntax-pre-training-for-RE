#!/bin/bash

EXP_PATH=<PATH TO EXPERIMENT FOLDER>

DATA_PATH=./data/crossre_data
PRE_TRAIN_DATA_PATH=./data/pre_train_syntax_data

SEEDS=( 4012 5096 8824 8257 9908 )

DOMAIN_TRAIN="ai" # "literature" "music" "news" "politics" "science"
DOMAIN_TEST=( "ai" "literature" "music" "news" "politics" "science" )

# iterate over seeds
for rs in "${!SEEDS[@]}"; do
  echo "Experiment on random seed ${SEEDS[$rs]}."

  # iterate over test sets
  for d_t in "${!DOMAIN_TEST[@]}"; do

    exp_dir=$EXP_PATH/rs${SEEDS[$rs]}

    # check if experiment already exists
    if [ -f "$exp_dir/best.pt" ]; then
      echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
    # if experiment is new, train classifier
    else
      echo "Training model ${TASK} on random seed ${SEEDS[$rs]}."

      # train
      python3 main.py \
              --train_path "${DATA_PATH}/${DOMAIN_TRAIN}-train.json" \
              --dev_path "${DATA_PATH}/${DOMAIN_TRAIN}-dev.json" \
              --exp_path ${exp_dir} \
              --stilt_train "${PRE_TRAIN_DATA_PATH}/ai.json ${PRE_TRAIN_DATA_PATH}/literature.json ${PRE_TRAIN_DATA_PATH}/music.json ${PRE_TRAIN_DATA_PATH}/news.json ${PRE_TRAIN_DATA_PATH}/politics.json ${PRE_TRAIN_DATA_PATH}/science.json" \
              --seed ${SEEDS[$rs]}
    fi

    # check if prediction already exists
    if [ -f "$exp_dir/${DOMAIN_TEST[$d_t]}-test-pred.csv" ]; then
      echo "[Warning] Prediction '$exp_dir/${DOMAIN_TEST[$d_t]}-test-pred.csv' already exists. Not re-predicting."

    # if no prediction is available, run inference
    else
      # prediction
      python3 main.py \
              --train_path "${DATA_PATH}/${DOMAIN_TRAIN}-train.json" \
              --test_path "${DATA_PATH}/${DOMAIN_TEST[$d_t]}-test.json" \
              --exp_path ${exp_dir} \
              --seed ${SEEDS[$rs]} \
              --prediction_only
    fi

    # check if summary metric scores file already exists
    if ! [ -f "$EXP_PATH/summary-exps-test-${DOMAIN_TEST[$d_t]}.txt" ]; then
        echo "Train on ${DOMAIN_TRAIN}; Test on ${DOMAIN_TEST[$d_t]}" > $EXP_PATH/summary-exps-test-${DOMAIN_TEST[$d_t]}.txt
    fi

    # run evaluation
    python3 evaluate.py \
            --gold_path ${DATA_PATH}/${DOMAIN_TEST[$d_t]}-test.json \
            --pred_path ${exp_dir}/${DOMAIN_TEST[$d_t]}-test-pred.csv \
            --out_path ${exp_dir} \
            --summary_exps $EXP_PATH/summary-exps-test-${DOMAIN_TEST[$d_t]}.txt


  done
done