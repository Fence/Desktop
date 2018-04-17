for ex in 3 
do
    for db in cooking win2k wikihow
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --actionDB $db \
        --train_mode af \
        --ex_count $ex \
        --gpu_rate 0.25
    done
done
