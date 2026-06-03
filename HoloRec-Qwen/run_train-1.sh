export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Instruments
BASE_MODEL=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/Qwen3-1-7B # LLaMA
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET-daoubel-4xi32cu-4gpu/

torchrun --nproc_per_node=4 --master_port=3326  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --learning_rate 2e-4 \
    --epochs 6 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.xinyan32.epoch10000.alpha2e-2-beta1e-4.json\
    --wandb_run_name test\
    --temperature 0.8
