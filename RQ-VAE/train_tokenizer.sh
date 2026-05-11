python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data_path /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/data/Instruments/Instruments.emb-llama-td.npy\
  --alpha 0 \
  --beta 0.0001 \
  --cf_emb /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  --ckpt_dir /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE/checkpoint/