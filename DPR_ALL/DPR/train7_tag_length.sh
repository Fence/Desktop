#!/bin/bash
start_time=$(date +%s)
for tl in 2 5 10 25 50 75 100 125 150
do
    CUDA_VISIBLE_DEVICES=1 python main.py \
    --batch_size 32  \
    --epochs 4 \
    --actionDB 'tag_actions tag_actions1 tag_actions2 tag_actions3 tag_actions5 tag_actions6' \
    --max_text_num '64 52 33 54 35 43' \
    --test_text_num 8 \
    --char_emb_flag 0 \
    --tag_length $tl \
    --decay_rate 0.88 \
    --gpu_rate 0.20 \
    --load_weights "" \
    --save_weights_prefix '' \
    --result_dir "results/basic_results/tb012356/tag_length/rp_new_bs32_ce0_tl$tl" \
    --computer_id 2
done 
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
