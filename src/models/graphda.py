"""
GraphDA for MMRec — full two-stage pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyTorch implementation of **Graph Collaborative Signals Denoising & Augmentation (GraphDA)**
(SIGIR 2023; Fan *et al.*) adapted to **MMRec** with the *pretrain → distill → retrain*
workflow baked in, so you can reproduce paper results in a single model class.

Pipeline
--------
1) **Pre-train backbone** (LightGCN) for `pretrain_epochs` on the raw UI graph.
2) **Enhanced-UI distillation**: keep top-`distill_userK` interactions per user and
   top-`distill_itemK` per item (scored by pretrained dot-product), optionally filtering
   by `distill_thres`.
3) **Add UU & II correlations**: build user–user and item–item KNN edges from the
   balanced UI matrix, thresholding by `uuii_thres`.
4) **Retrain** with LightGCN propagation on the **augmented adjacency** (UI + UU + II),
   using standard BPR + L2.

Notes
-----
* This file assumes MMRec-style dataset & trainer utilities: `dataset.inter_matrix()`,
  `GeneralRecommender`, and `BPRLoss`/`EmbLoss`.
* Distillation is done **once** at model init; set `pretrain_epochs=0` to skip pretraining
  (will use random embeddings to score, not recommended for replication).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class GraphDA(GeneralRecommender):
    """GraphDA – adjacency denoising & augmentation over LightGCN backbone."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # === Data ===
        self.ui_raw: sp.coo_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # === Backbone & training hyper-parameters ===
        self.latent_dim = int(config["embedding_size"])
        self.n_layers = int(config["n_layers"])           # retrain depth
        self.learning_rate = float(config["learning_rate"])
        self.reg_weight = float(config["reg_weight"])

        # === Distillation / augmentation knobs (paper & repo flags) ===
        self.pretrain_epochs = int(config["pretrain_epochs"])
        self.pretrain_lr = float(config["pretrain_lr"])
        self.pretrain_layers = int(config["pretrain_layers"])   # LightGCN layers for pretrain
        self.pretrain_weight_decay = float(config["pretrain_decay"])
        self.pretrain_keep_prob = float(config["pretrain_keep_prob"])

        self.distill_layers = int(config["distill_layers"])
        self.distill_userK = int(config["distill_userK"])
        self.distill_itemK = int(config["distill_itemK"])
        self.distill_thres = float(config["distill_thres"])   # -1 to disable

        self.distill_uuK = int(config["distill_uuK"])
        self.distill_iiK = int(config["distill_iiK"])
        self.uuii_thres = float(config["uuii_thres"])       # -1 to disable

        # EASE regularization for UU/II (optional, see YAML)
        self.uu_lambda = float(config["uu_lambda"])
        self.ii_lambda = float(config["ii_lambda"])
        self.sim_mode = str(config["sim_mode"])  # {'cosine','ease'}

        # === Parameters ===
        self.user_emb = nn.Parameter(torch.randn(self.n_users, self.latent_dim) * 0.01)
        self.item_emb = nn.Parameter(torch.randn(self.n_items, self.latent_dim) * 0.01)

        # Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # === Build augmented adjacency once ===
        # Optionally pretrain a small LightGCN to score edges
        if self.pretrain_epochs > 0:
            u_pre, i_pre = self._pretrain_lightgcn()
        else:
            u_pre = self.user_emb.detach().cpu().numpy()
            i_pre = self.item_emb.detach().cpu().numpy()
        # Distill UI and add UU/II
        self.aug_adj = self._build_augmented_graph(u_pre, i_pre).to(self.device)

    # ------------------------------------------------------------------
    #   GraphDA — build augmented adjacency
    # ------------------------------------------------------------------
    def _pretrain_lightgcn(self) -> Tuple[np.ndarray, np.ndarray]:
        """A tiny LightGCN pretrain to obtain scoring embeddings for distillation."""
        # Local copy of embeddings for pretrain
        u = torch.randn(self.n_users, self.latent_dim) * 0.01
        v = torch.randn(self.n_items, self.latent_dim) * 0.01
        u = u.to(self.device)
        v = v.to(self.device)

        adj = self._build_norm_adj(self.ui_raw).to(self.device)
        opt = torch.optim.Adam([u.requires_grad_(), v.requires_grad_()], lr=self.pretrain_lr, weight_decay=self.pretrain_weight_decay)

        # Pre-compute positives list per user for quick sampling
        R = self.ui_raw.tocsr()
        pos_lists = [R.getrow(uid).indices for uid in range(self.n_users)]

        def propagate(uE, vE):
            allE = torch.cat([uE, vE], dim=0)
            embs = [allE]
            for _ in range(self.pretrain_layers):
                allE = torch.sparse.mm(adj, allE)
                embs.append(allE)
            out = torch.stack(embs, dim=1).mean(dim=1)
            return torch.split(out, [self.n_users, self.n_items])

        bpr = BPRLoss()
        for _ in range(self.pretrain_epochs):
            # simple one pass over users (uniform negatives)
            users = torch.arange(self.n_users, device=self.device)
            # sample one positive per user
            pos_items = []
            for uid in range(self.n_users):
                items = pos_lists[uid]
                if items.size == 0:
                    pos_items.append(0)
                else:
                    pos_items.append(np.random.choice(items))
            pos_items = torch.tensor(pos_items, device=self.device)
            # negative sampling
            neg_items = torch.randint(0, self.n_items, (self.n_users,), device=self.device)
            for uid in range(self.n_users):
                if pos_lists[uid].size == 0:
                    continue
                while neg_items[uid].item() in pos_lists[uid]:
                    neg_items[uid] = torch.randint(0, self.n_items, (1,), device=self.device)

            u_all, i_all = propagate(u, v)
            u_e = u_all[users]
            pos_e = i_all[pos_items]
            neg_e = i_all[neg_items]
            loss = bpr((u_e * pos_e).sum(-1), (u_e * neg_e).sum(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # return numpy arrays
        with torch.no_grad():
            u_all, i_all = propagate(u, v)
        return u_all.detach().cpu().numpy(), i_all.detach().cpu().numpy()

    def _build_norm_adj(self, ui: sp.coo_matrix) -> torch.sparse.FloatTensor:
        """D^{-1/2}(A+I)D^{-1/2} for a bipartite UI matrix."""
        rows = np.concatenate([ui.row, ui.col + self.n_users])
        cols = np.concatenate([ui.col + self.n_users, ui.row])
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        A.setdiag(1.0)
        deg = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(deg, -0.5)
        L = sp.diags(d_inv_sqrt) @ A @ sp.diags(d_inv_sqrt)
        L = L.tocoo()
        idx = torch.LongTensor([L.row, L.col])
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, vals, torch.Size(L.shape))

    def _distill_ui(self, u_pre: np.ndarray, i_pre: np.ndarray) -> sp.coo_matrix:
        """Enhanced-UI via top-K within each user and item neighborhood.
        Scores are dot-products of **pretrained** embeddings.
        """
        R = self.ui_raw.tocsr()
        user_rows, item_cols, vals = [], [], []

        # per-user topK over existing positives
        for uid in range(self.n_users):
            items = R.indices[R.indptr[uid]:R.indptr[uid+1]]
            if items.size == 0:
                continue
            scores = u_pre[uid] @ i_pre[items].T
            if self.distill_thres >= 0:
                mask = scores >= self.distill_thres
                items = items[mask]
                scores = scores[mask]
                if items.size == 0:
                    continue
            k = min(self.distill_userK, items.size)
            topk_idx = np.argpartition(-scores, k-1)[:k]
            sel_items = items[topk_idx]
            user_rows.extend([uid] * len(sel_items))
            item_cols.extend(sel_items.tolist())
            vals.extend([1.0] * len(sel_items))

        # per-item topK over existing positives
        RT = R.T
        for iid in range(self.n_items):
            users = RT.indices[RT.indptr[iid]:RT.indptr[iid+1]]
            if users.size == 0:
                continue
            scores = i_pre[iid] @ u_pre[users].T
            if self.distill_thres >= 0:
                mask = scores >= self.distill_thres
                users = users[mask]
                scores = scores[mask]
                if users.size == 0:
                    continue
            k = min(self.distill_itemK, users.size)
            topk_idx = np.argpartition(-scores, k-1)[:k]
            sel_users = users[topk_idx]
            user_rows.extend(sel_users.tolist())
            item_cols.extend([iid] * len(sel_users))
            vals.extend([1.0] * len(sel_users))

        ui = sp.coo_matrix((np.array(vals, np.float32), (np.array(user_rows), np.array(item_cols))),
                           shape=(self.n_users, self.n_items))
        ui.sum_duplicates()
        return ui

    def _cosine_knn(self, mat: sp.csr_matrix, k: int, thres: float):
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        row_norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1 + 1e-8
        rows, cols = [], []
        for r in range(mat.shape[0]):
            row = mat.getrow(r)
            sims = row @ mat.T
            sims = sims.toarray().ravel()
            sims[r] = 0.0
            sims = sims / (row_norms[r] * row_norms + 1e-8)
            if thres >= 0:
                sims[sims < thres] = -1e9
            k_eff = min(k, sims.size-1)
            idx = np.argpartition(-sims, k_eff)[:k_eff]
            rows.extend([r] * len(idx))
            cols.extend(idx.tolist())
        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)

    def _ease_weights(self, X: sp.csr_matrix, lam: float) -> sp.csr_matrix:
        """Compute EASE-r item–item weights (or user–user if X is transposed).
        NOTE: this is a dense operation O(n^2) memory – use with caution.
        Returns a sparse matrix with row-normalized weights.
        """
        G = (X.T @ X).astype(np.float64).toarray()
        n = G.shape[0]
        G[np.arange(n), np.arange(n)] += lam
        P = np.linalg.inv(G) @ (X.T @ X).toarray()  # simplified closed-form
        np.fill_diagonal(P, 0.0)
        row_norm = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
        S = P / row_norm
        return sp.csr_matrix(S)

    def _build_augmented_graph(self, u_pre: np.ndarray, i_pre: np.ndarray) -> torch.sparse.FloatTensor:
        # 1) Enhanced-UI via distillation
        ui = self._distill_ui(u_pre, i_pre).tocsr()

        # 2) UU & II KNN edges
        if self.sim_mode == "ease":
            ii_sim = self._ease_weights(ui, self.ii_lambda)
            uu_sim = self._ease_weights(ui.T, self.uu_lambda)
        else:
            uu_r, uu_c = self._cosine_knn(ui, self.distill_uuK, self.uuii_thres)
            ii_r, ii_c = self._cosine_knn(ui.T, self.distill_iiK, self.uuii_thres)
            uu_sim = sp.coo_matrix((np.ones_like(uu_r, dtype=np.float32), (uu_r, uu_c)), shape=(self.n_users, self.n_users)).tocsr()
            ii_sim = sp.coo_matrix((np.ones_like(ii_r, dtype=np.float32), (ii_r, ii_c)), shape=(self.n_items, self.n_items)).tocsr()

        # 3) Assemble full (UI + UU + II) adjacency
        ui = ui.tocoo()
        ui_rows = ui.row
        ui_cols = ui.col + self.n_users

        uu = uu_sim.tocoo()
        ii = ii_sim.tocoo()

        row_idx = np.concatenate([ui_rows, ui_cols, uu.row, ii.row + self.n_users])
        col_idx = np.concatenate([ui_cols, ui_rows, uu.col, ii.col + self.n_users])
        data = np.ones_like(row_idx, dtype=np.float32)

        A = sp.coo_matrix((data, (row_idx, col_idx)), shape=(self.n_nodes, self.n_nodes))
        A.setdiag(1.0)
        # symmetric normalization
        deg = np.array(A.sum(axis=1)).flatten() + 1e-7
        d_inv = np.power(deg, -0.5)
        L = sp.diags(d_inv) @ A @ sp.diags(d_inv)
        L = L.tocoo()
        idx = torch.LongTensor([L.row, L.col])
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, vals, torch.Size(L.shape))

    # ------------------------------------------------------------------
    #   LightGCN over augmented graph
    # ------------------------------------------------------------------
    def forward(self):
        all_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.aug_adj, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=1).mean(dim=1)
        user_all, item_all = torch.split(out, [self.n_users, self.n_items])
        return user_all, item_all

    def calculate_loss(self, interaction):
        user, pos, neg = interaction[0], interaction[1], interaction[2]
        user_all, item_all = self.forward()
        u_e = user_all[user]
        pos_e = item_all[pos]
        neg_e = item_all[neg]
        mf = self.bpr_loss((u_e * pos_e).sum(-1), (u_e * neg_e).sum(-1))
        reg = self.reg_loss(self.user_emb[user], self.item_emb[pos], self.item_emb[neg])
        return mf + self.reg_weight * reg

    def full_sort_predict(self, interaction):
        uid = interaction[0]
        user_all, item_all = self.forward()
        return torch.matmul(user_all[uid], item_all.T)
