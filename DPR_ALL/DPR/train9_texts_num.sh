#!/bin/bash
start_time=$(date +%s)
for pr in 0.75
do
    python main.py \
    --target_steps 5 \
    --batch_size 32  \
    --epochs 50 \
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
    --result_dir "results/basic_results/tb012356/train_epoch50" \
    --computer_id 1
done
end_time=$(date +%s)
echo -e "\n\nTotal time cost: $(($end_time - $start_time))s \n\n"
