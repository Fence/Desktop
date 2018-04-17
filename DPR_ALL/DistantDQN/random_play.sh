for db in wikihow cooking win2k
do
    for am in eas af
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --epochs 1 \
        --random_play 1 \
        --agent_mode $am \
        --actionDB $db \
        --result_dir "random_play" 
    done
done
