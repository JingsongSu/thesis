export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Instruments
OUTPUT_DIR=./ckpt/$DATASET-base/

torchrun --nproc_per_node=2 --master_port=4314 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .tw32.json \
    --coarse_index_file .tw32.json \
    --temperature 0.7 \
    --coarse_loss_weight 5 \
    --fine_loss_weight 1.0 \
    --coarse_align_weight 200.0
