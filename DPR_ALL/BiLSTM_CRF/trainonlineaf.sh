for st in 5 10 15 
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --actionDB wikihow \
    --train_mode af \
    --max_expend_num 2 \
    --epochs 5 \
    --online_text_num 18 \
    --online_text_step $st \
    --gpu_rate 0.20 \
    --load_weights "wikihow" \
    --result_dir "wikihow_online_step"$st 
done
