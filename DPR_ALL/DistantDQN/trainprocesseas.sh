nohup bash -c \
'for uar in 0 1 
do
    for pr in 0 1
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --use_act_rate $uar \
        --action_rate 0.05 \
        --log_train_process 1 \
        --online_text_num 2 \
        --agent_mode eas \
        --epochs 2 \
        --optimizer adam \
        --priority $pr \
        --positive_rate 0.9 \
        --gpu_rate 0.15 \
        --actionDB cooking \
        --result_dir "log_process_uar"$uar"_pr"$pr 
    done
done' \
> cooking_process_eas.out 2>&1 & 
