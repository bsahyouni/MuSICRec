# -*- coding: utf-8 -*-
"""
SimGCL for MMRec
~~~~~~~~~~~~~~~~
PyTorch implementation of **SimGCL – Simplifying Graph Contrastive Learning for Recommendation**
(SIGIR 2022, Yu *et al.*) adapted to the MMRec framework.

Core idea (paper §3):
• **No graph augmentation needed** – Instead of dropping edges/nodes, SimGCL creates two views by
  injecting small **sign-preserving random noise** into the LightGCN embeddings.
• **InfoNCE loss** between the clean and noisy views regularises the model, improving recommendation.

Notation
--------
* ``tau`` – temperature of InfoNCE.
* ``ssl_reg`` – weight of the self-supervised loss.
* ``eps`` – scale of the random noise (ξ in Eq. 4 of the paper).

Implementation choices here follow the authors’ **SELFRec** reference code.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class SimGCL(GeneralRecommender):
    """Simplified Graph Contrastive Learning recommender."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # —— Dataset ——
        self.inter_mat: sp.coo_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # —— Hyper‑parameters ——
        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        # SSL
        self.ssl_reg = config["ssl_reg"]
        self.tau = config["ssl_temp"]
        self.eps = config["eps"] # noise scale ε

        # —— Parameters ——
        self.user_emb = nn.Parameter(torch.randn(self.n_users, self.latent_dim) * 0.01)
        self.item_emb = nn.Parameter(torch.randn(self.n_items, self.latent_dim) * 0.01)

        # —— Graph ——
        self.norm_adj = self._build_norm_adj().to(self.device)

        # —— Losses ——
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    # ------------------------------------------------------------------
    #   Graph helper
    # ------------------------------------------------------------------
    def _build_norm_adj(self):
        ui = self.inter_mat
        rows = np.concatenate([ui.row, ui.col + self.n_users])
        cols = np.concatenate([ui.col + self.n_users, ui.row])
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        A.setdiag(1.0)
        deg = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(deg, -0.5)
        D_inv = sp.diags(d_inv_sqrt)
        L = D_inv @ A @ D_inv
        L = L.tocoo()
        indices = torch.LongTensor([L.row, L.col])
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, vals, torch.Size(L.shape))

    # ------------------------------------------------------------------
    #   Propagation (LightGCN backbone)
    # ------------------------------------------------------------------
    def _get_ego_emb(self):
        return torch.cat([self.user_emb, self.item_emb], dim=0)

    def lightgcn_forward(self):
        emb = self._get_ego_emb()
        embs = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            embs.append(emb)
        out = torch.stack(embs, dim=1).mean(dim=1)  # mean pooling
        users, items = torch.split(out, [self.n_users, self.n_items])
        return users, items, out

    # ------------------------------------------------------------------
    #   SimGCL self‑supervised loss
    # ------------------------------------------------------------------
    def _simgcl_noise(self, emb: torch.Tensor) -> torch.Tensor:
        """Generate noisy view: ẽ = e + ε · sign(e) · ξ where ξ ~ U(-1,1)."""
        rand = torch.rand_like(emb)
        sign = (rand > 0.5).float() * 2 - 1  # ±1
        noise = F.normalize(sign, dim=1)  # keep scale 1 in expectation
        return emb + self.eps * noise

    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss between two views (node‑wise)."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / self.tau  # (N,N)
        sim_exp = torch.exp(sim)
        pos = torch.diag(sim_exp)
        denom = sim_exp.sum(dim=1)
        loss = -torch.log(pos / denom)
        return loss.mean()

    # ------------------------------------------------------------------
    #   Training loss
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_all, item_all, all_emb = self.lightgcn_forward()

        # BPR
        u_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]
        mf_loss = self.bpr_loss((u_e * pos_e).sum(dim=1), (u_e * neg_e).sum(dim=1))
        reg_loss = self.reg_loss(self.user_emb[users], self.item_emb[pos_items], self.item_emb[neg_items])

        # SimGCL contrastive loss (on full graph to match paper).
        noisy_emb = self._simgcl_noise(all_emb)
        cl_loss = self._info_nce(all_emb, noisy_emb)

        return mf_loss + self.reg_weight * reg_loss + self.ssl_reg * cl_loss

    # ------------------------------------------------------------------
    #   Full-sort prediction
    # ------------------------------------------------------------------
    def full_sort_predict(self, interaction):
        user_idx = interaction[0]
        user_all, item_all, _ = self.lightgcn_forward()
        return torch.matmul(user_all[user_idx], item_all.T)
