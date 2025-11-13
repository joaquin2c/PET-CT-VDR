#!/usr/bin/env bash
model="$1"
gpu="$2"
#python train_models.py -d stanford -b swin3D -m ct -gpu "$gpu" -l focal -p ../../../data/PET-CT/data/"$model" -e "$model"_focal
#python train_models.py -d stanford -b swin3D -m pet -gpu "$gpu" -l focal -p ../../../data/PET-CT/data/"$model" -e "$model"_focal
#python train_models.py -d stanford -b swin3D -m petct -gpu "$gpu" -l focal -p ../../../data/PET-CT/data/"$model" -e "$model"_focal
#python train_models.py -d stanford -b swin3D -m petct -gpu "$gpu" -l crossmodal -p ../../../data/PET-CT/data/"$model" -e "$model"_cross

python avg_kfold_metrics.py -p ../models -m "$model"_focal >> "../log_metrics/metrics_$model.log"
python avg_kfold_metrics.py -p ../models -m "$model"_cross >> "../log_metrics/metrics_$model.log"
