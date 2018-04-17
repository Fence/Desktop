#!/bin/bash
start_time=$(date +%s)
for dvs in 10 20 50 100 150 200 300 500
do
    for bs in 32
    do
        python main.py \
        --dynamic_vocab_size $dvs \
        --batch_size $bs \
        --train_repeat 1 \
        --tag_dim 1 \
        --gpu_rate 0.24 \
        --load_weights "" \
        --save_weights_prefix '' \
        --result_dir "results/win2k/dvs"$dvs"_train_repeat1_bs"$bs \
        --computer_id 1
    done
done 
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
