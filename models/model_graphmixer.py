from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch_geometric.nn import GATv2Conv
from torch.nn.init import kaiming_normal_
from timm.models.layers import DropPath, trunc_normal_

from models.model_utils import *
from models.self_attention import *

class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim, bias=False),
            torch.nn.Dropout(dropout)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        return self.net(x)

class GraphMixerBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., dropout = 0.1, drop_path=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = GATv2Conv(dim, dim, dropout=0.1)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mix = MLP(dim=dim, hidden_dim=mlp_hidden_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, edge_index):
        x = x + self.token_mix(self.norm1(x), edge_index)
        x = x + self.channel_mix(x)

        return x

class GraphMixer_Surv(torch.nn.Module):
    def __init__(self, input_dim=2227, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=512, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, omic_sizes=[], n_classes=4):  #[89, 334, 534, 471, 1509, 482]
        super(GraphMixer_Surv, self).__init__()
        self.use_edges = use_edges
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        self.embedding = nn.Sequential(*[nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        self.graphmixer_blocks = nn.ModuleList([])
        for level in range(1, self.num_layers+1):
            self.graphmixer_blocks.append(GraphMixerBlock(hidden_dim, level))

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])

        ### Constructing Genomic SNN
        if omic_sizes:
            hidden = [256, hidden_dim]  #hidden_dim
            self.omic_sizes = omic_sizes
            sig_networks = []
            for input_dim in self.omic_sizes:
                fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
                for i, _ in enumerate(hidden[1:]):
                    fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
                sig_networks.append(nn.Sequential(*fc_omic))
            self.sig_networks = nn.ModuleList(sig_networks)

            self.coattn = MultiHeadAttention(in_features=hidden_dim, head_num=8)
        
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self,  **kwargs):
        data = kwargs['x_path']
        x_omic = kwargs['x_omic']
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        x = self.embedding(data.x)
        for graphmixer_block in self.graphmixer_blocks:
            x = graphmixer_block(x, edge_index)  

        if x_omic:
            h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
            h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)
            x = x.unsqueeze(1)

            h_omic_bag = h_omic_bag.transpose(0, 1)
            x = x.transpose(0, 1)
            x, G_coattn = self.coattn(h_omic_bag, x, x)
            x = x.squeeze(0)
        else:
            # print('WIS only')
            G_coattn = None
        h_path = x  
        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path)
        logits  = self.classifier(h) #.unsqueeze(0) # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, G_coattn, logits
