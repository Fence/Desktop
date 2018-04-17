#!/bin/bash
start_time=$(date +%s)
for bs in 32
do
    for pr in 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
    do
        python main.py \
        --batch_size $bs  \
        --epochs 3 \
        --start_epoch 0 \
        --positive_rate $pr \
        --actionDB 'tag_actions tag_actions1 tag_actions2 tag_actions3 tag_actions5 tag_actions6' \
        --max_text_num '64 52 33 54 35 43' \
        --test_text_num 8 \
        --char_emb_flag 0 \
        --tag_length 50 \
        --decay_rate 0.88 \
        --gpu_rate 0.40 \
        --load_weights "" \
        --save_weights_prefix '' \
        --result_dir "results/basic_results/tb012356/pr"$pr"_rp_new35k_ce0_bs32" \
        --computer_id 1
    done
done 
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
