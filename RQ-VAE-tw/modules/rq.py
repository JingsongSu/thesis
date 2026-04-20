import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 1,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])


    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def vq_ini(self, x):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):

            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x, labels, use_sk=True):
        all_losses = []
        all_indices1 = []
        all_indices2 = []
        all_indices3 = []

        # x_q = 0
        # residual = x
        x_q1 = 0
        x_q2 = 0
        x_q3 = 0

        residual1 = x[:, :8]
        residual2 = x[:, :16]
        residual3 = x[:, :]

        residual1_copy = x[:, :8]
        residual2_copy = x[:, :16]
        residual3_copy = x[:, :]

        for idx, quantizer in enumerate(self.vq_layers):
            # label = labels[str(idx)]
            
            # x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk)
            x_res1, loss1, indices1 = quantizer(residual1, idx, input_dim=8)
            x_res2, loss2, indices2 = quantizer(residual2, idx, input_dim=16)
            x_res3, loss3, indices3 = quantizer(residual3, idx, input_dim=32)
            
            residual1 = residual1 - x_res1
            residual2 = residual2 - x_res2
            residual3 = residual3 - x_res3
            
            x_q1 = x_q1 + x_res1
            x_q2 = x_q2 + x_res2
            x_q3 = x_q3 + x_res3

            all_losses.append((loss1 + loss2 + loss3) / 3)
            all_indices1.append(indices1)
            all_indices2.append(indices2)
            all_indices3.append(indices3)
        
        loss_q1 = F.mse_loss(x_q1, residual1_copy)
        loss_q2 = F.mse_loss(x_q2, residual2_copy)
        loss_q3 = F.mse_loss(x_q3, residual3_copy)

        mean_recon_loss = (loss_q1 + loss_q2 + loss_q3) / 3
        mean_losses = torch.stack(all_losses).mean()
        all_indices1 = torch.stack(all_indices1, dim=-1)
        all_indices2 = torch.stack(all_indices2, dim=-1)
        all_indices3 = torch.stack(all_indices3, dim=-1)
        # all_indices = torch.stack(all_indices, dim=-1)

        return x_q1, x_q2, x_q3, mean_losses, all_indices1, all_indices2, all_indices3, mean_recon_loss