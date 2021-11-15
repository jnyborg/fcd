#!/bin/bash

cd ..

DEVICE=$1
RUNS=3

if [ $DEVICE == "cuda:0" ]; then
  # CAM
  for i in $(seq 1 $RUNS)
  do
    python cam_main.py --mode train --cam_method cam --experiment_name ResNet50CAM_$i --device $DEVICE
    python cam_main.py --mode test --cam_method cam --experiment_name ResNet50CAM_$i --device $DEVICE
  done

  # GradCAM
  for i in $(seq 1 $RUNS)
  do
    python cam_main.py --mode train --cam_method gradcam --experiment_name ResNet50GradCAM_$i --device $DEVICE
    python cam_main.py --mode test --cam_method gradcam --experiment_name ResNet50GradCAM_$i --device $DEVICE
  done
elif [ $DEVICE == "cuda:1" ]; then
  # GradCAM++
  for i in $(seq 1 $RUNS)
  do
    python cam_main.py --mode train --cam_method gradcampp --experiment_name ResNet50GradCAMpp_$i --device $DEVICE
    python cam_main.py --mode test --cam_method gradcampp --experiment_name ResNet50GradCAMpp_$i --device $DEVICE
  done

  # U-CAM
  for i in $(seq 1 $RUNS)
  do
    python cam_main.py --mode train --cam_method ucam --experiment_name UCAM_$i --device $DEVICE
    python cam_main.py --mode test --cam_method ucam --experiment_name UCAM_$i --device $DEVICE
  done

else
    echo "Available arguments are cuda:0, cuda:1."
    exit 1
fi
