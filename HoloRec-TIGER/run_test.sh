DATASET=Instruments
DATA_PATH=../data
CKPT_PATH=./ckpt/$DATASET-base/checkpoint-18447

RESULTS_FILE=./results/$DATASET/all-interleave.json
SAVE_PRED_TXT=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-TIGER-new/predictions.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=2214 test.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --test_batch_size 32 \
  --num_beams 20 \
  --save_pred_txt $SAVE_PRED_TXT \
  --test_prompt_ids 0 \
  --index_file .tw32.json \
  --coarse_index_file .tw32.json
