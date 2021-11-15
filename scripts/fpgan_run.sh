#!/bin/bash

cd ..

DEVICE=$1

if [ $DEVICE == "cuda:0" ]; then
  python main.py --mode train --dataset L8Biome --image_size 128 --c_dim 1 --batch_size 16 \
                 --experiment_name FixedPointGAN_1 --device $DEVICE

  python main.py --mode train --dataset L8Biome --image_size 128 --c_dim 1 --batch_size 16 \
                 --experiment_name FixedPointGAN_3 --device $DEVICE

elif [ $DEVICE == "cuda:1" ]; then
  python main.py --mode train --dataset L8Biome --image_size 128 --c_dim 1 --batch_size 16 \
                 --experiment_name FixedPointGAN_2 --device $DEVICE
else
    echo "Available arguments are cuda:0, cuda:1."
    exit 1
fi




