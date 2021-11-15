#!/bin/bash

cd ..

DEVICE=$1
RUNS=3

if [ $DEVICE == "cuda:0" ]; then
  # FCD+100%: Train U-Net with all available labels
  for i in $(seq 1 $RUNS)
  do
    python supervised_main.py --mode train --batch_size 64 \
                              --classifier_head False \
                              --experiment_name FullySupervised_Manual_NoClassifier_$i \
                              --device $DEVICE
  done

  # FCD+Pre: Pre-train U-Net encoder on image-level labels
  for i in $(seq 1 $RUNS)
  do
    python supervised_main.py --mode train --batch_size 64 \
                              --train_encoder_only True \
                              --experiment_name PretrainEncoder_L8Biome_$i \
                              --device $DEVICE
  done

  # FCD+Pre1%: Fine-tune previous model with 1% of pixel-level labels
  for i in $(seq 1 $RUNS)
  do
    python supervised_main.py --mode train --batch_size 64 \
                              --keep_ratio 0.01 \
                              --lr 1e-5 \
                              --freeze_encoder True \
                              --encoder_weights outputs/PretrainEncoder_L8Biome_$i/models/l8biome_resnet34.pt \
                              --experiment_name FineTune1Pct_L8BiomeEncoder_$i \
                              --device $DEVICE
  done

elif [ $DEVICE == "cuda:1" ]; then
  # FCD+: Train U-Net on Fixed-Point GAN masks.
  for i in $(seq 1 $RUNS)
  do
    python supervised_main.py --mode train --batch_size 64 \
                              --train_mask_file generated_mask.tif \
                              --classifier_head True \
                              --experiment_name FullySupervised_Generated_$i \
                              --device $DEVICE
  done

  # FCD+1%: Fine-tune previous model on 1% of pixel-level labels.
  for i in $(seq 1 $RUNS)
  do
    python supervised_main.py --mode train --batch_size 64 \
                              --keep_ratio 0.01 \
                              --lr 1e-5 \
                              --freeze_encoder True \
                              --model_weights outputs/FullySupervised_Generated_$i/models/best.pt \
                              --experiment_name FineTune1Pct_FullySupervised_Generated_$i \
                              --device $DEVICE
  done

  # FCD+Rand1%: Train model from random initialization with 1% of pixel-level labels.
  for i in $(seq 1 $RUNS)
  do
    python supervised_main.py --mode train --batch_size 64 \
                              --keep_ratio 0.01 \
                              --lr 1e-5 \
                              --classifier_head True \
                              --experiment_name FineTune1Pct_RandomInit_$i \
                              --device $DEVICE
  done
else
    echo "Available arguments are cuda:0, cuda:1."
    exit 1
fi



