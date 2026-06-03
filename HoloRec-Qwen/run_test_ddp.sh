export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET=Instruments
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/results/$DATASET/ddp.json
BASE_MODEL=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/Qwen3-1-7B # LLaMA

torchrun --nproc_per_node=8 --master_port=4324 test_ddp.py \
    --ckpt_path  /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-LC-Rec/ckpt/Instruments-daoubel-4xi32cu-4gpu/checkpoint-20600 \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --save_simple_results  \
    --index_file .index.xinyan32.epoch10000.alpha2e-2-beta1e-4.json

