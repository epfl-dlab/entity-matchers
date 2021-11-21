#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/scratch/huber/Entity-Match-Project/deepmatcher

folds[0]=1
folds[1]=2
folds[2]=3
folds[3]=4
folds[4]=5

folder_output=$1
folder_original_datasets=$2
folder_deepmatcher_datasets=$3

for fold in ${folds[*]}; do
  echo run fold ${fold}
  mkdir -p ${folder_output}/EN-DE-15K
  time CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
      --dataset_deepmatcher ${folder_deepmatcher_datasets}/XRealEA_Translated/deepmatcher/EN-DE-15K-5NEIGH-SUM \
      --dataset_original ${folder_original_datasets}/XRealEA_Translated/EN-DE-15K \
      --folds ${fold} > ${folder_output}/EN-DE-15K/train_EN-DE-15K-5NEIGH-SUM_fold_${fold}.log
  echo run EN-DE
  mkdir -p ${folder_output}/EN-FR-15K
  time CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
    --dataset_deepmatcher ${folder_deepmatcher_datasets}/XRealEA_Translated/deepmatcher/EN-FR-15K-5NEIGH-SUM \
    --dataset_original ${folder_original_datasets}/XRealEA_Translated/EN-FR-15K \
    --folds ${fold} > ${folder_output}/EN-FR-15K/train_EN-FR-15K-5NEIGH-SUM_fold_${fold}.log
  echo run EN-FR
  mkdir -p ${folder_output}/EN-JA-15K
  time CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
    --dataset_deepmatcher ${folder_deepmatcher_datasets}/dataset/XRealEA_Translated/deepmatcher/EN-JA-15K-5NEIGH-SUM \
    --dataset_original ${folder_original_datasets}/XRealEA_Translated/EN-JA-15K \
    --folds ${fold} > ${folder_output}/EN-JA-15K/train_EN-JA-15K-5NEIGH-SUM_fold_${fold}.log
  echo run EN-JA
  mkdir -p ${folder_output}/DB-YG-15K
  time CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
    --dataset_deepmatcher ${folder_deepmatcher_datasets}/dataset/RealEA/deepmatcher/DB-YG-15K-5NEIGH-SUM \
    --dataset_original ${folder_original_datasets}/RealEA/DB-YG-15K \
    --folds ${fold} > ${folder_output}/DB-YG-15K/train_DB-YG-15K-5NEIGH-SUM_fold_${fold}.log
  echo run DB-YG
  mkdir -p ${folder_output}/DB-WD-15K
  time CUDA_VISIBLE_DEVICES=1 python run_experiment.py \
    --dataset_deepmatcher ${folder_deepmatcher_datasets}/dataset/RealEA/deepmatcher/DB-WD-15K-5NEIGH-SUM \
    --dataset_original ${folder_original_datasets}/RealEA/DB-WD-15K \
    --folds ${fold} > ${folder_output}/DB-WD-15K/train_DB-WD-15K-5NEIGH-SUM_fold_${fold}.log
  echo run DB-WD
done
