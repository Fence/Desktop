for db in wikihow
do
    python main.py \
    --agent_mode 'multi' \
    --actionDB $db \
    --result_dir 'pipeline_result'
done
