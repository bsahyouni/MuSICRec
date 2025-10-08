# -*- coding: utf-8 -*-
"""
NCL for MMRec
~~~~~~~~~~~~~
PyTorch implementation of **Neighborhood‑enriched Contrastive Learning (NCL)**
(WWW 2022, Lin *et al.*) adapted to the MMRec framework.

Key ideas (see paper §3):
* **GraphCF backbone** – LightGCN (linear propagation on the user–item graph).
* **Neighborhood‑enriched contrastive loss** – combines:
  - *Layer contrast* (node vs. its representation after `hyper_layers*2` propagations).
  - *Prototype contrast* (node vs. centroid of its K‑means cluster), weighted by `proto_reg`.
* **Self‑supervised loss weight** – `ssl_reg · (user_loss + α · item_loss)`.

The code closely follows the official RecBole implementation but is refactored to
plug directly into **MMRec** (subclass of `GeneralRecommender`).
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss_gpu  # GPU K‑Means for fast prototype updates
except ImportError:  # fallback: scikit‑learn k‑means
    faiss = None
    from sklearn.cluster import KMeans

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class NCL(GeneralRecommender):
    """Neighborhood‑enriched Contrastive Learning (Lin *et al.*, WWW ’22)."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---------- dataset ---------- #
        self.inter_mat: sp.coo_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # ---------- hyper‑parameters ---------- #
        self.latent_dim: int = config["embedding_size"]
        self.n_layers: int = config["n_layers"]
        self.reg_weight: float = config["reg_weight"]  # L2

        # SSL specific
        self.ssl_temp: float = config["ssl_temp"]
        self.ssl_reg: float = config["ssl_reg"]
        self.alpha: float = config["alpha"]  # weight on item part of ssl loss
        self.hyper_layers: int = config["hyper_layers"]  # step size for contrastive view

        # ProtoNCE
        self.proto_reg: float = config["proto_reg"]
        self.k_clusters: int = config["num_clusters"]
        self.cluster_refresh: int = config["cluster_refresh"]  # epoch interval

        # runtime counters
        self._epoch_cnt: int = 0

        # ---------- parameters ---------- #
        initializer = nn.init.xavier_uniform_
        self.user_emb = nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim)))
        self.item_emb = nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))

        # ---------- graph ---------- #
        self.norm_adj = self._build_norm_adj().to(self.device)

        # ---------- loss ---------- #
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # ---------- clustering state ---------- #
        self.user_centroids = None  # [k, d]
        self.item_centroids = None
        self.user2cluster = None  # [n_users]
        self.item2cluster = None

    # ------------------------------------------------------------------
    #   Graph utilities
    # ------------------------------------------------------------------
    def _build_norm_adj(self) -> torch.sparse.FloatTensor:
        """Construct D^(−1/2) (A + I) D^(−1/2) sparse tensor."""
        ui = self.inter_mat
        # bipartite to symmetric adjacency
        R = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        # user‑to‑item and item‑to‑user
        R._update(dict(zip(zip(ui.row, ui.col + self.n_users), [1.0] * ui.nnz)))
        R._update(dict(zip(zip(ui.col + self.n_users, ui.row), [1.0] * ui.nnz)))
        # add self‑loops
        R.setdiag(1.0)
        deg = np.array(R.sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(deg, -0.5)
        D_inv = sp.diags(d_inv_sqrt)
        L = D_inv @ R @ D_inv
        L = L.tocoo()
        indices = torch.LongTensor([L.row, L.col])
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, data, torch.Size(L.shape))

    def _get_ego_emb(self):
        return torch.cat([self.user_emb, self.item_emb], dim=0)

    # ------------------------------------------------------------------
    #   Forward propagation (LightGCN)
    # ------------------------------------------------------------------
    def forward(self):
        all_emb = self._get_ego_emb()
        embs = [all_emb]
        for _ in range(max(self.n_layers, self.hyper_layers * 2)):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        lightgcn_emb = torch.stack(embs[: self.n_layers + 1], dim=1).mean(dim=1)
        user_all, item_all = torch.split(lightgcn_emb, [self.n_users, self.n_items])
        return user_all, item_all, embs  # embs includes layerwise representations

    # ------------------------------------------------------------------
    #   Contrastive components
    # ------------------------------------------------------------------
    def _protoNCE_loss(self, node_emb: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Prototype‑level contrastive loss (Eq. 7)."""
        if self.user_centroids is None or self.item_centroids is None:
            return torch.tensor(0.0, device=self.device)
        # Users
        user_vec = F.normalize(node_emb[users], dim=1)
        user_cent = self.user_centroids[self.user2cluster[users]]  # [B, d]
        pos_user = torch.exp((user_vec * user_cent).sum(dim=1) / self.ssl_temp)
        ttl_user = torch.exp(user_vec @ self.user_centroids.T / self.ssl_temp).sum(dim=1)
        loss_user = -torch.log(pos_user / ttl_user).sum()
        # Items
        item_vec = F.normalize(node_emb[items + self.n_users], dim=1)
        item_cent = self.item_centroids[self.item2cluster[items]]
        pos_item = torch.exp((item_vec * item_cent).sum(dim=1) / self.ssl_temp)
        ttl_item = torch.exp(item_vec @ self.item_centroids.T / self.ssl_temp).sum(dim=1)
        loss_item = -torch.log(pos_item / ttl_item).sum()
        return self.proto_reg * (loss_user + loss_item)

    def _layer_ssl_loss(self, emb_t: torch.Tensor, emb_0: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Neighborhood‑enriched layer contrast (Eq. 5)."""
        # Split current & previous embeddings into user/item
        u_cur, i_cur = torch.split(emb_t, [self.n_users, self.n_items])
        u_prev, i_prev = torch.split(emb_0, [self.n_users, self.n_items])
        # Users
        u1 = F.normalize(u_cur[users], dim=1)
        u2 = F.normalize(u_prev[users], dim=1)
        all_prev_u = F.normalize(u_prev, dim=1)
        pos_user = torch.exp((u1 * u2).sum(dim=1) / self.ssl_temp)
        ttl_user = torch.exp(u1 @ all_prev_u.T / self.ssl_temp).sum(dim=1)
        ssl_user = -torch.log(pos_user / ttl_user).sum()
        # Items
        i1 = F.normalize(i_cur[items], dim=1)
        i2 = F.normalize(i_prev[items], dim=1)
        all_prev_i = F.normalize(i_prev, dim=1)
        pos_item = torch.exp((i1 * i2).sum(dim=1) / self.ssl_temp)
        ttl_item = torch.exp(i1 @ all_prev_i.T / self.ssl_temp).sum(dim=1)
        ssl_item = -torch.log(pos_item / ttl_item).sum()
        return self.ssl_reg * (ssl_user + self.alpha * ssl_item)

    # ------------------------------------------------------------------
    #   K‑Means updates
    # ------------------------------------------------------------------
    def _run_kmeans(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if faiss is not None:
            kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k_clusters, gpu=True)
            kmeans.train(data)
            centroids = torch.from_numpy(kmeans.centroids).to(self.device)
            _, I = kmeans.index.search(data, 1)
            assignments = torch.from_numpy(I).long().squeeze(1).to(self.device)
        else:
            km = KMeans(n_clusters=self.k_clusters, n_init=10, random_state=42).fit(data)
            centroids = torch.from_numpy(km.cluster_centers_).to(self.device)
            assignments = torch.from_numpy(km.labels_).long().to(self.device)
        centroids = F.normalize(centroids, dim=1)
        return centroids, assignments

    def _refresh_prototypes(self):
        with torch.no_grad():
            user_np = self.user_emb.detach().cpu().numpy()
            item_np = self.item_emb.detach().cpu().numpy()
        self.user_centroids, self.user2cluster = self._run_kmeans(user_np)
        self.item_centroids, self.item2cluster = self._run_kmeans(item_np)

    # ------------------------------------------------------------------
    #   Training / inference
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        # refresh prototypes periodically
        if (self._epoch_cnt % self.cluster_refresh == 0) and (self._epoch_cnt == 0 or self.proto_reg > 0):
            self._refresh_prototypes()
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all, item_all, embs = self.forward()
        # BPR loss
        u_e = user_all[user]
        pos_e = item_all[pos_item]
        neg_e = item_all[neg_item]
        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        mf_loss = self.bpr_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(self.user_emb[user], self.item_emb[pos_item], self.item_emb[neg_item])

        # SSL losses
        layer_loss = self._layer_ssl_loss(embs[self.hyper_layers * 2], embs[0], user, pos_item)
        proto_loss = self._protoNCE_loss(embs[0], user, pos_item)

        total = mf_loss + self.reg_weight * reg_loss + layer_loss + proto_loss
        return total

    def full_sort_predict(self, interaction):
        user_idx = interaction[0]
        user_all, item_all, _ = self.forward()
        scores = torch.matmul(user_all[user_idx], item_all.T)
        return scores

    # ------------------------------------------------------------------
    #   MMRec trainer hooks
    # ------------------------------------------------------------------
    def on_epoch_end(self):
        """Called by trainer after each epoch."""
        self._epoch_cnt += 1