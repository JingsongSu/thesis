import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 1,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list # 每个量化器的码本大小，[256,256,256]
        self.e_dim = e_dim # 每个量化器码本嵌入向量的维度
        self.num_quantizers = len(n_e_list) # 几层码本
        self.kmeans_init = kmeans_init # 是否使用KMeans初始化码本
        self.kmeans_iters = kmeans_iters # KMeans的最大迭代次数
        self.sk_epsilons = sk_epsilons # 每个量化器在 Sinkhorn 方法中的 epsilon 参数，用于正则化
        self.sk_iters = sk_iters # Sinkhorn 方法的最大迭代次数
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ]) # 4×(256,32)
        self.vq_layers2 = nn.ModuleList([VectorQuantizer(n_e, int(e_dim/4), beta=beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ]) # 4×(256,16)
        

    # 获取所有码本
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers: # 遍历每个量化器，调用其 get_codebook() 方法
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook) # 将所有码本堆叠成一个张量，形状为 (num_quantizers, n_e, e_dim)
        return torch.stack(all_codebook) # 一个张量，包含所有量化器的码本
    
    # 量化器初始化
    def vq_ini(self, x):
        x_q = 0 # 存储量化后的表征
        residual = x # 初始化 residual 为输入 x
        for idx, quantizer in enumerate(self.vq_layers): # 遍历每个量化器，调用其 vq_init 方法对残差进行量化初始化

            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res # 每次量化后，更新残差为 residual - x_res
            x_q = x_q + x_res # 不断相加x_res是量化后的表征
        
        x_q = 0 # 存储量化后的表征
        residual = x[:,8:16] # 初始化 residual 为输入 x
        for idx, quantizer in enumerate(self.vq_layers2): # 遍历每个量化器，调用其 vq_init 方法对残差进行量化初始化

            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res # 每次量化后，更新残差为 residual - x_res
            x_q = x_q + x_res # 不断相加x_res是量化后的表征

    def forward(self, x, labels, use_sk=True): # 前向传播逐层量化 x:(64,32);label:4(256)
        all_losses = []
        all_indices = []

        x_q = 0 # 初始化 x_q 为 0，用于存储量化后的表示
        residual = x # 初始化 residual 为输入 x，表示当前的残差

        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)] # 使用当前 residual 和对应标签 label 进行量化 256个每个位置0-9的聚类编号
            
            x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk) # 获取量化后的表示 x_res、量化损失 loss 和量化索引 indices 第0层emb:(64,32);label:4(256)
            residual = residual - x_res # 更新残差为 residual - x_res
            x_q = x_q + x_res # 将 x_res 加入量化结果 x_q

            all_losses.append(loss) 
            all_indices.append(indices) # 将每层的损失和索引存入列表
        
        x_q2 = 0 
        residual = x[:,8:16]

        for layer1, layer2 in zip(self.vq_layers, self.vq_layers2):
            # 获取 vq_layers 中嵌入权重的前4维，并赋值给 vq_layers2
            layer2.embedding.weight.data = layer1.embedding.weight.data[:,8:16]

        for idx, quantizer in enumerate(self.vq_layers2):
            label = labels[str(idx+4)] # 使用当前 residual 和对应标签 label 进行量化 256个每个位置0-9的聚类编号
            
            x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk) # 获取量化后的表示 x_res、量化损失 loss 和量化索引 indices 第0层emb:(64,32);label:4(256)
            residual = residual - x_res # 更新残差为 residual - x_res
            x_q2 = x_q2 + x_res # 将 x_res 加入量化结果 x_q
            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean() # 所有量化器损失的平均值，用于优化目标
        all_indices = torch.stack(all_indices, dim=-1) # 所有量化器的索引，形状为 (batch_size, num_quantizers)

        return x_q, mean_losses, all_indices # 量化后的表示、所有量化器的平均量化损失、所有量化器的索引



# import torch
# import torch.nn as nn

# from .vq import VectorQuantizer


# class ResidualVectorQuantizer(nn.Module):

#     def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 1,
#                  kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
#         super().__init__()
#         self.n_e_list = n_e_list
#         self.e_dim = e_dim
#         self.num_quantizers = len(n_e_list)
#         self.kmeans_init = kmeans_init
#         self.kmeans_iters = kmeans_iters
#         self.sk_epsilons = sk_epsilons
#         self.sk_iters = sk_iters
#         self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=beta,
#                                                         kmeans_init = self.kmeans_init,
#                                                         kmeans_iters = self.kmeans_iters,
#                                                         sk_epsilon=sk_epsilon,
#                                                         sk_iters=sk_iters)
#                                         for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])


#     def get_codebook(self):
#         all_codebook = []
#         for quantizer in self.vq_layers:
#             codebook = quantizer.get_codebook()
#             all_codebook.append(codebook)
#         return torch.stack(all_codebook)
    
#     def vq_ini(self, x):
#         x_q = 0
#         residual = x
#         for idx, quantizer in enumerate(self.vq_layers):

#             x_res = quantizer.vq_init(residual, use_sk=True)
#             residual = residual - x_res
#             x_q = x_q + x_res

#     def forward(self, x, labels, use_sk=True):
#         all_losses = []
#         all_indices = []

#         x_q = 0
#         residual = x

#         for idx, quantizer in enumerate(self.vq_layers):
#             label = labels[str(idx)]
            
#             x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk)
#             residual = residual - x_res
#             x_q = x_q + x_res

#             all_losses.append(loss)
#             all_indices.append(indices)

#         mean_losses = torch.stack(all_losses).mean()
#         all_indices = torch.stack(all_indices, dim=-1)

#         return x_q, mean_losses, all_indices