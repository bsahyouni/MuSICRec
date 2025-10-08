"""
LightGCL for MMRec
~~~~~~~~~~~~~~~~~~~
PyTorch implementation of **LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation**
(ICLR 2023; Cai, Huang, Xia, Ren) adapted to MMRec.

LightGCL = LightGCN backbone (local dependency) + an SVD-guided global view, with
**InfoNCE** between the two views in addition to the standard **BPR** loss.

Key args (mapped to the official repo names):
- ``ssl_reg``  (λ₁ in the paper / ``--lambda1`` in the repo): weight on contrastive loss
- ``reg_weight`` (λ₂ / ``--lambda2``): L2 regularization weight
- ``ssl_temp`` (τ / ``--temp``): contrastive temperature
- ``edge_dropout`` (``--dropout``): edge dropout on the main graph
- ``svd_rank`` (``--q``): truncated SVD rank used for the global view

References:
- Paper & method (SVD view; local–global contrastive learning).
- Official PyTorch implementation and CLI flags.

The implementation avoids materialising the dense SVD-reconstructed graph by computing
messages as (US)(VᵀE) or (VS)(UᵀE) (see repo README Q&A on complexity), which keeps the
cost proportional to *(I+J)* instead of *(I·J)*.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class LightGCL(GeneralRecommender):
    """LightGCL – LightGCN + SVD-guided contrastive augmentation."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---------------- Data ---------------- #
        self.inter_mat: sp.coo_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items

        # ---------------- Hyper-parameters ---------------- #
        self.latent_dim = int(config["embedding_size"])
        self.n_layers = int(config["n_layers"])
        self.reg_weight = float(config["reg_weight"])

        # CL head
        self.ssl_reg = float(config["ssl_reg"])
        self.ssl_temp = float(config["ssl_temp"])
        self.edge_dropout = float(config["edge_dropout"])
        self.svd_rank = int(config["svd_rank"])

        # ---------------- Parameters ---------------- #
        self.user_emb = nn.Parameter(torch.randn(self.n_users, self.latent_dim) * 0.01)
        self.item_emb = nn.Parameter(torch.randn(self.n_items, self.latent_dim) * 0.01)

        # ---------------- Graphs ---------------- #
        self.norm_adj = self._build_norm_adj().to(self.device)
        # Pre-compute SVD factors on the (binary) interaction matrix (I×J)
        self.US, self.VS, self.UT, self.VT = self._build_svd_factors(self.inter_mat, self.svd_rank)
        self.US = self.US.to(self.device)
        self.VS = self.VS.to(self.device)
        self.UT = self.UT.to(self.device)
        self.VT = self.VT.to(self.device)

        # ---------------- Losses ---------------- #
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    # ==============================================================
    # Graph utilities
    # ==============================================================
    def _build_norm_adj(self) -> torch.sparse.FloatTensor:
        """Return \hat{A} = D^{-1/2}(A+I)D^{-1/2} as torch.sparse."""
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
        idx = torch.LongTensor([L.row, L.col])
        vals = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, vals, torch.Size(L.shape))

    def _sparse_dropout(self, mat: torch.sparse.FloatTensor, p: float) -> torch.sparse.FloatTensor:
        """Edge dropout on a COO sparse tensor (keep-prob = 1-p)."""
        if p <= 0.0:
            return mat
        idx = mat._indices()
        val = mat._values()
        nnz = val.size(0)
        mask = (torch.rand(nnz, device=val.device) > p)
        idx = idx[:, mask]
        val = val[mask] / (1.0 - p)  # rescale to keep expectation
        return torch.sparse_coo_tensor(idx, val, mat.shape).coalesce()

    def _build_svd_factors(self, ui: sp.coo_matrix, q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute truncated SVD of the (binary) I×J interaction matrix.
        Returns (US, VS, U^T, V^T) for efficient message passing.
        """
        R = ui.tocsr()
        # Centering / normalisation left as-is to match public code; use implicit feedback (0/1)
        u, s, vt = svds(R.asfptype(), k=q, return_singular_vectors=True)
        # Order singular values descending
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]
        # Build factors
        US = u @ np.diag(s)          # (I, q)
        V = vt.T                     # (J, q)
        VS = V @ np.diag(s)          # (J, q)
        UT = u.T                     # (q, I)
        VT = vt                      # (q, J)
        # Convert to torch
        return (
            torch.from_numpy(US).float(),
            torch.from_numpy(VS).float(),
            torch.from_numpy(UT).float(),
            torch.from_numpy(VT).float(),
        )

    # ==============================================================
    # Propagation (two views)
    # ==============================================================
    def _lightgcn_propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main view: LightGCN with optional edge-dropout, sum over layers."""
        all_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        embs = [all_emb]
        adj = self.norm_adj
        for _ in range(self.n_layers):
            A = self._sparse_dropout(adj, self.edge_dropout)
            all_emb = torch.sparse.mm(A, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=1).sum(dim=1)  # paper uses sum
        users, items = torch.split(out, [self.n_users, self.n_items])
        return users, items

    def _svd_propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Global view: use low-rank factors to simulate message passing.
        For users:  (US) @ (Vᵀ @ E_v) ;  For items: (VS) @ (Uᵀ @ E_u)
        Accumulate across L layers with identity activation (as in LightGCN).
        """
        u = self.user_emb
        v = self.item_emb
        u_acc = u
        v_acc = v
        for _ in range(self.n_layers):
            # users
            tmp_i = torch.matmul(self.VT, v)        # (q, d)
            u = torch.matmul(self.US, tmp_i)        # (I, d)
            # items
            tmp_u = torch.matmul(self.UT, u_acc)    # (q, d)
            v = torch.matmul(self.VS, tmp_u)        # (J, d)
            u_acc = u_acc + u
            v_acc = v_acc + v
        return u_acc, v_acc

    # ==============================================================
    # Losses & prediction
    # ==============================================================
    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / tau
        sim_exp = torch.exp(sim)
        pos = torch.diag(sim_exp)
        denom = sim_exp.sum(dim=1)
        loss = -torch.log(pos / denom)
        return loss.mean()

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos = interaction[1]
        neg = interaction[2]

        u_main, i_main = self._lightgcn_propagate()
        u_svd, i_svd = self._svd_propagate()

        # BPR on main view
        u_e = u_main[users]
        pos_e = i_main[pos]
        neg_e = i_main[neg]
        mf = self.bpr_loss((u_e * pos_e).sum(dim=1), (u_e * neg_e).sum(dim=1))
        reg = self.reg_loss(self.user_emb[users], self.item_emb[pos], self.item_emb[neg])

        # Local–global contrast (users + items)
        cl_u = self._info_nce(u_main, u_svd, self.ssl_temp)
        cl_i = self._info_nce(i_main, i_svd, self.ssl_temp)
        cl = cl_u + cl_i

        return mf + self.reg_weight * reg + self.ssl_reg * cl

    def full_sort_predict(self, interaction):
        user_idx = interaction[0]
        u_main, i_main = self._lightgcn_propagate()
        return torch.matmul(u_main[user_idx], i_main.T)
