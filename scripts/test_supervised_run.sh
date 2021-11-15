#!/bin/bash

cd ..

RUNS=3

for i in $(seq 1 $RUNS)
do
  python supervised_main.py --mode test --batch_size 64 \
                            --experiment_name FullySupervised_Manual_$i
done

for i in $(seq 1 $RUNS)
do
  python supervised_main.py --mode test --batch_size 64 \
                            --experiment_name FineTune1Pct_L8BiomeEncoder_$i
done

for i in $(seq 1 $RUNS)
do
  python supervised_main.py --mode test --batch_size 64 \
                            --experiment_name FullySupervised_Generated_$i
done

for i in $(seq 1 $RUNS)
do
  python supervised_main.py --mode test --batch_size 64 \
                            --experiment_name FineTune1Pct_FullySupervised_Generated_$i
done

for i in $(seq 1 $RUNS)
do
  python supervised_main.py --mode test --batch_size 64 \
                            --experiment_name FineTune1Pct_RandomInit_$i
done
