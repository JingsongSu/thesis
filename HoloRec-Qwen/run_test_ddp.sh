export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

DATASET=Instruments
DATA_PATH=../data

CKPT_PATH=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/ckpt/Instruments-ceshi-4gpu/checkpoint-10300

RESULTS_FILE=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/results/$DATASET/ddp.json

BASE_MODEL=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/Qwen3-1-7B

torchrun --nproc_per_node=4 --master_port=4324 test_ddp.py \
    --ckpt_path "$CKPT_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --results_file "$RESULTS_FILE" \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --save_simple_results \
    --index_file .tw32.json \
    --coarse_index_file .tw8.json \
    --interleaved_inference \
    --interleaved_temperature 0.8
