for men in 2 4 8 
do
    for db in win2k 
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --actionDB $db \
        --train_mode eas \
        --op_count 2 \
        --ex_count 2 \
        --max_expend_num $men \
        --gpu_rate 0.40 \
        --result_dir "exc2_men"$men
    done
done
