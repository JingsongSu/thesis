export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Instruments
BASE_MODEL=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/Qwen3-1-7B
DATA_PATH=../data

OUTPUT_DIR=./ckpt/$DATASET-ceshi-4gpu/

torchrun --nproc_per_node=4 --master_port=3326 lora_finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --learning_rate 2e-4 \
    --epochs 6 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .tw32.json \
    --coarse_index_file .tw8.json \
    --wandb_run_name Instruments-interleaved-tw32tw8 \
    --temperature 0.8
