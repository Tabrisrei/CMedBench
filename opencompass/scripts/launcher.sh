config='experiments/template_config/origin_Meta-Llama-3-8B.py' 
result_dir='path/to/result' 
log_dir='path/to/log'
# mkdir for the folder of log_dir
mkdir -p $(dirname ${log_dir}) 

model_mode='pretrain'
export PYTHONPATH=llmc:$PYTHONPATH
nohup \
python run.py \
    opencompass/configs/${config} \
    --mode 'all' \
    --work-dir ${result_dir} \
    > ${log_dir} 2>&1 &


