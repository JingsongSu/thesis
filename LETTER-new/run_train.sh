export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Instruments
OUTPUT_DIR=./ckpt/$DATASET-32su16su-v3-05cora1fine10align-20early-2gpu-2tidu/

torchrun --nproc_per_node=2 --master_port=4314 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.xinyan32.epoch10000.alpha2e-2-beta1e-4.json \
    --coarse_index_file .index.xinyan16.epoch10000.alpha2e-2-beta1e-4.json \
    --temperature 0.7 \
    --coarse_loss_weight 0.5 \
    --fine_loss_weight 1.0 \
    --coarse_align_weight 10.0 \
    --curriculum_warmup_steps 0 
