import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from modules.rqvae import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--ckpt_path", type=str,default="/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE-tw/checkpoint/epoch_9999_collision_-2.5644_model.pth", help='root path')


    return parser.parse_args()

args_setting = parse_args()

ckpt_path = args_setting.ckpt_path

device = torch.device("cpu")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]

print(args.data_path)

data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
all_docs = []
prefix = ["{}","{}","{}"]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    # x = data.cpu().detach().numpy()
    # data = self.embedding.weight.cpu().detach().numpy()
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {"0":[],"1":[],"2":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]

from collections import defaultdict
sid_to_sku1 = defaultdict(list)
sid_to_sku2 = defaultdict(list)
sid_to_sku3 = defaultdict(list)

for d, doc_ids in tqdm(data_loader):
    doc_ids = doc_ids.tolist()
    d = d.to(device)
    
    indices1, indices2, indices3 = model.get_indices(d, labels, use_sk=False)
    indices1 = indices1.view(-1, indices1.shape[-1]).cpu().numpy()
    indices2 = indices2.view(-1, indices2.shape[-1]).cpu().numpy()
    indices3 = indices3.view(-1, indices3.shape[-1]).cpu().numpy()

    for doc_id, index in zip(doc_ids, indices1):
        code = []
        for ind in index:
            if isinstance(ind, np.ndarray):
                code.append(str(int(ind[0])))
            else:
                code.append(str(int(ind)))
        sid_str = "-".join(code)
        sid_to_sku1[sid_str].append(str(doc_id))

    for doc_id, index in zip(doc_ids, indices2):
        code = []
        for ind in index:
            if isinstance(ind, np.ndarray):
                code.append(str(int(ind[0])))
            else:
                code.append(str(int(ind)))
        sid_str = "-".join(code)
        sid_to_sku2[sid_str].append(str(doc_id))

    for doc_id, index in zip(doc_ids, indices3):
        code = []
        for ind in index:
            if isinstance(ind, np.ndarray):
                code.append(str(int(ind[0])))
            else:
                code.append(str(int(ind)))
        sid_str = "-".join(code)
        sid_to_sku3[sid_str].append(str(doc_id))


output_file1 = "/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE-tw/sid/nq_sid_tw1_rqvae.json"
output_file2 = "/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE-tw/sid/nq_sid_tw2_rqvae.json"
output_file3 = "/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE-tw/sid/nq_sid_tw3_rqvae.json"

# 保存结果
with open(output_file1, "w", encoding="utf-8") as file:
    json.dump(sid_to_sku1, file, ensure_ascii=False, indent=4)


with open(output_file2, "w", encoding="utf-8") as file:
    json.dump(sid_to_sku2, file, ensure_ascii=False, indent=4)


with open(output_file3, "w", encoding="utf-8") as file:
    json.dump(sid_to_sku3, file, ensure_ascii=False, indent=4)