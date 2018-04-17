for tn in 0 1 3 5 10 15 20 25 30 35 40 45 50
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --use_act_rate 0 \
        --action_rate 0.05 \
        --gram_num 5 \
        --dqn_mode cnn \
        --agent_mode eas \
        --epochs 3 \
        --optimizer adam \
        --learning_rate 0.0025 \
        --batch_size 32 \
        --tag_dim 50 \
        --reward_assign 50.0 \
        --positive_rate 0.9 \
        --gpu_rate 0.15 \
        --actionDB wikihow \
        --online_text_num $tn \
        --load_weights "weights/online_test/cooking/eas/fold0.h5" \
        --result_dir "online_tn"$tn"_pr0.90_ep2_uar0" 
done
