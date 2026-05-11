python ./RQ-VAE/main.py \
  --device cuda:2 \
  --data_path /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/data/Yelp/Yelp.emb-qwen3-8B-td.npy\
  --alpha 0 \
  --beta 0.0001 \
  --cf_emb /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE/ckpt/Yelp-32d-sasrec.pt\
  --ckpt_dir /home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE/checkpoint/
