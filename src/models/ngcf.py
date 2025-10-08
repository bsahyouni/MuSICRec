# -*- coding: utf-8 -*-
"""
NGCF for MMRec
~~~~~~~~~~~~~~
PyTorch implementation of **Neural Graph Collaborative Filtering (NGCF)**
(SIGIR 2019, Wang *et al.*) adapted for the MMRec framework.

Reference implementation sources:
* Original TensorFlow & PyTorch repos ([github.com](https://github.com/xiangwang1223/neural_graph_collaborative_filtering?utm_source=chatgpt.com))([github.com](https://github.com/huangtinglin/NGCF-PyTorch?utm_source=chatgpt.com))
* RecBole NGCF class (Apache‑licensed) ([recbole.io](https://recbole.io/docs/user_guide/model/general/ngcf.html))

Design highlights
-----------------
* **High‑order message passing** with trainable transformations `W1, W2` per layer.
* **LeakyReLU** activation (slope 0.2) after each layer.
* **Message dropout** (`mess_dropout`) applied to layer outputs (default 0.1).
* **Final embedding** is the **concatenation** of embeddings from all propagation layers, as in the paper (§3.4).
* **BPR + L2** objective identical to original experiment protocol.
"""

from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class NGCF(GeneralRecommender):
    """NGCF recommender adapted for MMRec.«"""  # noqa: D401

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # === Dataset ===
        self.inter_mat: sp.coo_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes: int = self.n_users + self.n_items

        # === Hyper‑parameters ===
        self.latent_dim: int = config["embedding_size"]
        self.n_layers: int = config["n_layers"]
        self.reg_weight: float = config["reg_weight"]
        # Message‑dropout probability per layer (list or scalar)
        md_default = [0.1] * self.n_layers
        self.mess_dropout: List[float] = config["mess_dropout"]
        if isinstance(self.mess_dropout, float):
            self.mess_dropout = [self.mess_dropout] * self.n_layers
        assert len(self.mess_dropout) == self.n_layers

        # === Parameters ===
        self.user_emb = nn.Parameter(torch.randn(self.n_users, self.latent_dim) * 0.01)
        self.item_emb = nn.Parameter(torch.randn(self.n_items, self.latent_dim) * 0.01)

        # Transformation matrices (W1 & W2 per layer)
        self.W1 = nn.ParameterList()
        self.W2 = nn.ParameterList()
        for _ in range(self.n_layers):
            self.W1.append(nn.Parameter(torch.empty(self.latent_dim, self.latent_dim)))
            self.W2.append(nn.Parameter(torch.empty(self.latent_dim, self.latent_dim)))
            nn.init.xavier_uniform_(self.W1[-1])
            nn.init.xavier_uniform_(self.W2[-1])

        # Activation and dropout modules
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.ModuleList([
            nn.Dropout(p=prob) for prob in self.mess_dropout
        ])

        # Build graph
        self.norm_adj = self._build_norm_adj().to(self.device)

        # Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    # ------------------------------------------------------------------
    #   Graph construction helpers
    # ------------------------------------------------------------------
    def _build_norm_adj(self) -> torch.sparse.FloatTensor:
        """Return D^(-1/2) (A + I) D^(-1/2) as torch sparse tensor."""
        ui = self.inter_mat
        # Bipartite adjacency -> symmetric
        rows = np.concatenate([ui.row, ui.col + self.n_users])
        cols = np.concatenate([ui.col + self.n_users, ui.row])
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        # Self‑loops
        A.setdiag(1.0)
        # Normalisation
        degree = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(degree, -0.5)
        D_inv = sp.diags(d_inv_sqrt)
        L = D_inv @ A @ D_inv
        L = L.tocoo()
        indices = torch.LongTensor([L.row, L.col])
        values = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(L.shape))

    # ------------------------------------------------------------------
    #   Propagation
    # ------------------------------------------------------------------
    def _get_ego_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_emb, self.item_emb], dim=0)

    def forward(self):
        ego_emb = self._get_ego_embeddings()
        all_emb = [ego_emb]
        emb = ego_emb
        for layer in range(self.n_layers):
            side_emb = torch.sparse.mm(self.norm_adj, emb)  # message passing
            # Feature transformation
            sum_emb = torch.matmul(side_emb, self.W1[layer])
            bi_emb = torch.mul(emb, side_emb)  # element‑wise product
            bi_emb = torch.matmul(bi_emb, self.W2[layer])
            emb = self.act(sum_emb + bi_emb)
            emb = self.dropout[layer](emb)
            all_emb.append(emb)
        # Concatenate layer‑wise embeddings (n_nodes, latent_dim * (n_layers+1))
        all_emb = torch.cat(all_emb, dim=1)
        user_all, item_all = torch.split(all_emb, [self.n_users, self.n_items])
        return user_all, item_all

    # ------------------------------------------------------------------
    #   Loss & prediction
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all, item_all = self.forward()
        u_e = user_all[user]
        pos_e = item_all[pos_item]
        neg_e = item_all[neg_item]
        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        mf_loss = self.bpr_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(self.user_emb[user], self.item_emb[pos_item], self.item_emb[neg_item])
        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user_idx = interaction[0]
        user_all, item_all = self.forward()
        scores = torch.matmul(user_all[user_idx], item_all.T)
        return scores
