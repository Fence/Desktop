#!/bin/bash
for nf in 8 16 32 64 128
do
    for ng in 4 5 6 7 8 9
    do
        CUDA_VISIBLE_DEVICES=4 python3 keras_train.py \
        --num_filter $nf \
        --num_gram $ng 
    done
done
