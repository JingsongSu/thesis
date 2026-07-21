export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,3

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET-fuxian-tw32tw8/

torchrun --nproc_per_node=2 --master_port=4311 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 400 \
    --index_file .tw32.json \
    --coarse_index_file .tw8.json \
    --temperature 1.0 \
    --coarse_loss_weight 0.5 \
    --fine_loss_weight 1.0 \
    --coarse_align_weight 30.0