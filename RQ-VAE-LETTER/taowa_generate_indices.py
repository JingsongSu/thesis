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
from models.rqvae import RQVAE
import argparse
import os

# 检查冲突相关函数

# 检查是否有编码冲突
def check_collision(all_indices_str):
    tot_item = len(all_indices_str) # 数据集中所有条目的总数
    tot_indice = len(set(all_indices_str.tolist())) # 去重后的条目数
    return tot_item==tot_indice # 如果总条目数和去重后条目数相等，说明没有冲突

# 获取每个编码的出现次数
def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int) # 创建一个默认值为 0 的字典
    for index in all_indices_str:
        indices_count[index] += 1 # 统计每个编码的出现次数
    return indices_count # 返回字典，键为编码，值为出现次数

# 获取冲突条目
def get_collision_item(all_indices_str):
    index2id = {} # 用于存储每个编码对应的条目索引
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i) # 将同一编码的条目索引存储到列表中

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1: # 如果一个编码对应的条目数大于 1，则认为有冲突
            collision_item_groups.append(index2id[index])

    return collision_item_groups # 返回冲突条目组

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE") 
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset') # 数据集名称
    parser.add_argument("--root_path", type=str,default="/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/RQ-VAE/checkpoint/alpha2e-2-beta1e-4/", help='root path') # 检查点文件路径
    parser.add_argument('--alpha', type=str, default='2e-2', help='cf loss weight') # # cf 损失权重
    parser.add_argument('--epoch', type=int, default='5000', help='epoch') # 训练轮次
    parser.add_argument('--checkpoint', type=str, default='filtered_model.pth', help='checkpoint name') # 检查点文件名
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight') # div 损失权重


    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
# ckpt_path = args_setting.root_path + f'Apr-09-2025_14-47-03/'+args_setting.checkpoint
ckpt_path = args_setting.root_path + args_setting.checkpoint

output_dir = f"/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/data/{dataset}/"
output_file = f"{dataset}.index.xinyan_ce32.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
output_file2 = f"{dataset}.index.xinyan_ce8.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
output_file = os.path.join(output_dir,output_file)
output_file2 = os.path.join(output_dir,output_file2)
device = torch.device("cuda:0") # 使用 GPU 0

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) # 加载检查点
args = ckpt["args"] # 获取训练的超参数
state_dict = ckpt["state_dict"] # 获取模型权重


# args.data_path = "/home/pod/shared-nvme/LETTER-main/data/Instruments/Instruments.emb-llama-td.npy"
data = EmbDataset(args.data_path) # 使用自定义数据集类加载嵌入数据

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

model.load_state_dict(state_dict,strict=False) # 加载权重
model = model.to(device) # 将模型加载到 GPU
model.eval() # 设置为评估模式（不需要梯度计算）
print(model)

# 创建数据加载器
data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices2 = []
all_indices_str = []
all_indices_str2 = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

# 定义约束性聚类方法
def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    # x = data.cpu().detach().numpy()
    # data = self.embedding.weight.cpu().detach().numpy()
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10) # 每个簇的最小大小
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels # 聚类中心；每个点对应的簇标签

labels = {"0":[],"1":[],"2":[], "3":[], "4":[], "5":[], "6":[], "7":[]} # 用于存储每层的聚类标签

embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers] # 提取每层的嵌入权重 4 个(256,32)
embs2  = [layer.embedding.weight.cpu().detach().numpy()[:,8:16] for layer in model.rq.vq_layers] # 提取每层的嵌入权重 4 个(256,32)

# 对每层嵌入进行聚类
for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb) # 对嵌入权重进行约束性 K-Means 聚类
    labels[str(idx)] = label # 将聚类标签存储到对应的层

for idx, emb in enumerate(embs2):
    centers, label = constrained_km(emb) # 对嵌入权重进行约束性 K-Means 聚类
    labels[str(idx+4)] = label # 将聚类标签存储到对应的层

# 遍历数据并生成编码
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1] # 提取输入数据和嵌入索引 (64,4096) (64)
    d = d.to(device) # 将数据加载到 GPU
    
    # indices = model.get_indices(d, use_sk=False)
    indices = model.get_indices(d, labels,use_sk=False) # 根据聚类标签获取编码  (64,4)->(64,8)
    indices1 = indices[:,:4]
    indices2 = indices[:,4:]
    indices1 = indices1.view(-1, indices1.shape[-1]).cpu().numpy() # 转换为 NumPy 数组  (64,8)
    indices2 = indices2.view(-1, indices2.shape[-1]).cpu().numpy() 

    for index in indices1:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind))) # 为每个维度生成带前缀的编码

        all_indices.append(code) # 保存编码
        all_indices_str.append(str(code)) # 保存编码的字符串形式
    # break

    for index in indices2:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind))) # 为每个维度生成带前缀的编码

        all_indices2.append(code) # 保存编码
        all_indices_str2.append(str(code)) # 保存编码的字符串形式
    # break
all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str) # 转换编码为数组
all_indices2 = np.array(all_indices2)
all_indices_str2 = np.array(all_indices_str2) # 转换编码为数组

# 调整量化层的参数
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0   # 设置前几层的 Soft K-Means epsilon 为 0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003 # 如果最后一层的 epsilon 为 0，则设置为 0.003，sk_epsilon 是 Soft K-Means 的一个超参数，用于控制量化时的松散程度

# model.rq.vq_layers[-1].sk_epsilon = 0.1

# 检查和解决冲突
tt = 0 # 冲突解决的迭代次数
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str): # 如果迭代次数达到上限或没有冲突，退出循环
        break

    collision_item_groups = get_collision_item(all_indices_str) # 获取冲突的条目组
    print(collision_item_groups) # 打印冲突组

    print(len(collision_item_groups)) # 打印冲突组的数量

    for collision_items in collision_item_groups:
        d = data[collision_items] # 获取冲突条目对应的数据
        d = d[0].to(device) # 加载到 GPU
        indices = model.get_indices(d, labels, use_sk=True) # 使用 Soft K-Means 获取新编码

        # indices = model.get_indices(d, use_sk=True)
        indices1 = indices[:,:4]
        indices1 = indices1.view(-1, indices1.shape[-1]).cpu().numpy() # 转换为 NumPy 数组
        for item, index in zip(collision_items, indices1):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind))) # 为新编码添加前缀

            all_indices[item] = code # 更新冲突条目的编码
            all_indices_str[item] = str(code) # 更新冲突条目的编码字符串
    tt += 1 # 增加迭代次数

tt = 0
while True:
    if tt >= 20 or check_collision(all_indices_str2): # 如果迭代次数达到上限或没有冲突，退出循环
        break

    collision_item_groups = get_collision_item(all_indices_str2) # 获取冲突的条目组
    print(collision_item_groups) # 打印冲突组

    print(len(collision_item_groups)) # 打印冲突组的数量

    for collision_items in collision_item_groups:
        d = data[collision_items] # 获取冲突条目对应的数据
        d = d[0].to(device) # 加载到 GPU
        indices = model.get_indices(d, labels, use_sk=True) # 使用 Soft K-Means 获取新编码

        # indices = model.get_indices(d, use_sk=True)
        indices2 = indices[:,4:]
        indices2 = indices2.view(-1, indices2.shape[-1]).cpu().numpy() # 转换为 NumPy 数组
        for item, index in zip(collision_items, indices2):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind))) # 为新编码添加前缀

            all_indices2[item] = code # 更新冲突条目的编码
            all_indices_str2[item] = str(code) # 更新冲突条目的编码字符串
    tt += 1 # 增加迭代次数

# 输出统计信息
print("All indices number: ",len(all_indices)) # 打印总编码数
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values())) # 打印最大冲突次数
print("Max number of conflicts2: ", max(get_indices_count(all_indices_str2).values())) # 打印最大冲突次数

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist())) # 计算去重后的编码数
print("Collision Rate",(tot_item-tot_indice)/tot_item) # 计算冲突率

# 保存编码到文件
all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices) # 将编码转换为字典形式，键为条目索引，值为编码

all_indices_dict2 = {}
for item, indices in enumerate(all_indices2.tolist()):
    all_indices_dict2[item] = list(indices) # 将编码转换为字典形式，键为条目索引，值为编码

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp) # 保存编码到 JSON 文件

with open(output_file2, 'w') as fp:
    json.dump(all_indices_dict2,fp) # 保存编码到 JSON 文件
