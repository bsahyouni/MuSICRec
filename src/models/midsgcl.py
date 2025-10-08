import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from copy import deepcopy
import math, random

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss

# ---- cache helpers ------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _edge_cache_path(n_subseq, th, metric):

    th_str = f"{th:.3f}"
    return os.path.join(_CACHE_DIR, f"ss_edges_{metric}_{n_subseq}_th{th_str}.npz")

def jaccard_sim(seq1, seq2):
    """
    Jaccard similarity of two (variable-length) subsequences.
    Treat each subsequence as a set of item IDs.
    """
    set1, set2 = set(seq1), set(seq2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return 0.0 if union == 0 else inter / union

#################################################
# 1) LEVENSHTEIN DISTANCE FUNCTION
#################################################
def levenshtein_distance(seq1, seq2):
    """
    Computes Levenshtein edit distance between two sequences (lists of item IDs).
    Returns the number of single-element edits (insert, delete, substitute)
    needed to transform seq1 into seq2.
    """
    n1, n2 = len(seq1), len(seq2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(n1 + 1):
        dp[i][0] = i
    for j in range(n2 + 1):
        dp[0][j] = j

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost  # substitution or match
            )
    return dp[n1][n2]

def build_subseq_tensor(subseq_map, max_len):
    # Build two tensors (N_subseq,L) once: item id pad and mask
    N = len(subseq_map)
    batch_items = torch.zeros(N, max_len, dtype=torch.long)
    batch_mask  = torch.zeros(N, max_len, dtype=torch.bool)
    for sid,(items) in enumerate(subseq_map.values()):
        items = items[:max_len]
        batch_items[sid, :len(items)] = torch.tensor(items)
        batch_mask[sid, :len(items)]  = 1
    return batch_items, batch_mask


#################################################
# 2) BUILD SUBSEQUENCE–SUBSEQUENCE + SUBSEQUENCE–ITEM GRAPH
#################################################
def build_subseq_graph_with_items(subseq_map, n_items, dist_threshold):
    """
    Builds a combined adjacency for:
      - Subsequence-Subsequence (SS) edges, weighted = 1 if edit distance <= dist_threshold
      - Subsequence-Item (SI) edges, weighted = 1 if item is in subsequence
    Returns a torch.sparse.FloatTensor of shape [n_s + n_items, n_s + n_items].

    subseq_map: dict {subseq_id: [item_ids]}
    n_items: total item count
    dist_threshold: maximum Levenshtein distance to connect two subsequences
    """

    print("correct cl file is running")
    # 1) Gather all subsequence IDs
    all_subseq_ids = sorted(subseq_map.keys())
    n_s = len(all_subseq_ids)

    row_indices, col_indices, data_vals = [], [], []

    # ---------- (A)  try load --------------------------------------------
    edge_file = _edge_cache_path(n_s, dist_threshold, "jacc")   # <<< metric tag
    if os.path.exists(edge_file):
        cached = np.load(edge_file)
        row_indices.extend(cached["row"])
        col_indices.extend(cached["col"])
        # data_vals.extend(np.ones_like(cached["row"], dtype=np.float32))
        data_vals.extend(cached["data"])

    else:
        # ---------- (B)  compute once, then save --------------------------
        for i in tqdm(range(n_s), desc="Jaccard S-S", unit="subseq"):
            seq1 = subseq_map[all_subseq_ids[i]]
            for j in range(i + 1, n_s):
                seq2 = subseq_map[all_subseq_ids[j]]
                sim = jaccard_sim(seq1, seq2)
                if sim >= dist_threshold:   # <<< Jaccard test
                    row_indices += [i, j]
                    col_indices += [j, i]
                    data_vals += [sim, sim]
        # np.savez_compressed(edge_file,
        #                     row=np.array(row_indices, dtype=np.int32),
        #                     col=np.array(col_indices, dtype=np.int32))
        np.savez_compressed(edge_file,
                            row=np.array(row_indices, dtype=np.int32),
                            col=np.array(col_indices, dtype=np.int32),
                            data=np.array(data_vals, dtype=np.float32))

    # # ---------- (A)  try load pre-computed edges --------------------------
    # edge_file = _edge_cache_path(n_s, dist_threshold)
    # if os.path.exists(edge_file):
    #     cached = np.load(edge_file)
    #     row_indices.extend(cached["row"])
    #     col_indices.extend(cached["col"])
    #     data_vals.extend(np.ones_like(cached["row"], dtype=np.float32))
    # else:
    #     # ---------- (B)  compute once, then save --------------------------
    #     for i in tqdm(range(n_s), desc="Levenshtein S-S", unit="subseq"):
    #         seq1 = subseq_map[all_subseq_ids[i]]
    #         for j in range(i + 1, n_s):
    #             seq2 = subseq_map[all_subseq_ids[j]]
    #             if levenshtein_distance(seq1, seq2) <= dist_threshold:
    #                 row_indices += [i, j]  # undirected
    #                 col_indices += [j, i]
    #     # persist for future runs
    #     np.savez_compressed(edge_file,
    #                         row=np.array(row_indices, dtype=np.int32),
    #                         col=np.array(col_indices, dtype=np.int32))
    # # ----------------------------------------------------------------------

    # ---------- S-I EDGES---------------
    offset = n_s
    for s_id, items in subseq_map.items():
        for it in items:
            row_indices += [s_id,         offset + it]
            col_indices += [offset + it,  s_id]
            data_vals   += [1.0, 1.0]

    # 4) Build scipy COO matrix
    assert len(row_indices) == len(col_indices) == len(data_vals), \
        f"row:{len(row_indices)}  col:{len(col_indices)}  data:{len(data_vals)}"
    size_total = n_s + n_items
    coo = sp.coo_matrix(
            (np.asarray(data_vals, dtype=np.float32),
                       (np.asarray(row_indices, dtype=np.int32),
                                  np.asarray(col_indices, dtype=np.int32))),
            shape = (size_total, size_total)
                         )

    # # ──--- diagnostics: proportion of S-S edges ---──
    # # both row & col < n_s   ⇒   endpoints are subsequences
    # ss_mask = (coo.row < n_s) & (coo.col < n_s)
    # ss_edges = ss_mask.sum()  # integer
    # all_edges = coo.nnz  # total #edges (symmetry included)
    # print(f"[INFO] dist_threshold={dist_threshold} → "
    #       f"{ss_edges:,}/{all_edges:,} edges "
    #       f"({ss_edges / all_edges:.2%}) are subsequence-to-subsequence.")
    # breakpoint()
    # # ────────────────────────────────────────────────

    # 5) Normalize adjacency (GCN-style: D^-1/2 * A * D^-1/2)
    row_sum = np.array(coo.sum(axis=1)).flatten() + 1e-7
    d_inv_sqrt = np.power(row_sum, -0.5)
    D = sp.diags(d_inv_sqrt)
    coo_norm = D @ coo @ D

    # Convert to PyTorch sparse
    coo_norm = coo_norm.tocoo()
    i = torch.LongTensor([coo_norm.row, coo_norm.col])
    v = torch.FloatTensor(coo_norm.data)
    return torch.sparse.FloatTensor(i, v, torch.Size([size_total, size_total]))


#################################################
# 3) SSG MODEL (SUBSEQUENCE–SUBSEQUENCE + USER–ITEM)
#################################################
class MIDSGCL(GeneralRecommender):
    r"""
    Model with:
      1) A user–item graph (norm_adj / masked_adj).
      2) A subsequence–subsequence + subsequence–item graph (ss_adj).
    Removes item–item adjacency from the old FREEDOM approach.
    """

    def __init__(self,
                 config,
                 dataset,
                 data_extra):
        super(MIDSGCL, self).__init__(config, dataset)

        # Basic settings
        self.embedding_dim = config['embedding_size']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.n_ui_layers = config['n_ui_layers']  # GNN layers for user–item
        self.n_ss_layers = config['n_ss_layers']  # GNN layers for subsequence–subsequence
        # self.reg_weight = config['reg_weight']  # regularization
        self.dropout = config['dropout']  # adjacency dropout
        self.device = config['device']
        self.beta = config['model_beta']
        self.alpha = config['model_alpha']
        self.use_user_seq_fusion = bool(config['use_user_seq_fusion'])
        self.fusion_norm = bool(config['fusion_norm'])
        self.tau = config['tau']
        self.uigraph_weight = 1.0  # weight factor for BPR loss
        self.lambda_seqcl = config['lambda_seqcl']
        self.lambda_itemcl = config['lambda_itemcl']
        self.lambda_userseqcl = config['lambda_userseqcl']

        # Prepare user–item adjacency
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None

        # user–item dropout edge info
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        self.max_len = 50

        self.pooling_type = config['pooling_type']
        if self.pooling_type == "addatt":
            self.att_id = AdditiveAttentionPool(self.embedding_dim, self.embedding_dim // 2, max_len=self.max_len).to(
                self.device)
            self.att_mm = AdditiveAttentionPool(self.embedding_dim, self.embedding_dim // 2, max_len=self.max_len).to(
                self.device)
        elif self.pooling_type == "sinus":
            self.att_id = SinusoidalPool(self.embedding_dim, max_len=self.max_len).to(self.device)
            self.att_mm = SinusoidalPool(self.embedding_dim, max_len=self.max_len).to(self.device)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        # --- NEW: Subsequence Embeddings + Subsequence Graph ---
        # We assume config or dataset can provide these:
        #  - n_subseq: total subsequence count
        #  - subseq_map: dict {subseq_id: [item_ids]}
        #  - dist_threshold: max Levenshtein distance for linking two subseqs

        self.n_subseq = data_extra.n_subseq
        self.subseq_map = data_extra.subseq_map
        self.dist_threshold = config['ss_dist_threshold']
        sub_items, sub_mask = build_subseq_tensor(self.subseq_map,
                                                            max_len=self.max_len)
        self.register_buffer("sub_items_buf", sub_items.to(self.device))  # (N_sub,L)
        self.register_buffer("sub_mask_buf", sub_mask.to(self.device))  # (N_sub,L)
        self.subseq_cache = nn.Embedding(self.n_subseq, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.subseq_cache.weight)

        dev0 = self.item_embedding.weight.device
        self.sub_items_buf = self.sub_items_buf.to(dev0)
        self.sub_mask_buf = self.sub_mask_buf.to(dev0)

        inv = [[] for _ in range(self.n_items)]
        for sid, items in self.subseq_map.items():
            for it in items:
                inv[it].append(sid)

        flat = []
        offset = [0]
        for lst in inv:  # ← the `inv` you already build
            flat.extend(lst)
            offset.append(len(flat))
        self.item2subseq_flat = torch.tensor(flat, dtype=torch.long,
                                             device=self.device)
        self.item2subseq_ptr = torch.tensor(offset, dtype=torch.long,
                                            device=self.device)
        # Build subsequence adjacency = SS + SI
        if self.beta ==0 and self.n_ss_layers==0:
            self.si_adj = None
        else:
            self.si_adj = self.get_subsequence_adj_mat().coalesce().to(self.device)

        self.refresh_mode = "lazy"
        self.n_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.mm_adj = None

        if self.v_feat is not None:
            print("visual features not none")
            v = torch.as_tensor(self.v_feat, dtype=torch.float32)
            self.image_embedding = nn.Embedding.from_pretrained(v, freeze=False).to(self.device)
            self.image_trs = nn.Linear(v.size(1), self.embedding_dim).to(self.device)

        if self.t_feat is not None:
            print("textual features not none")
            t = torch.as_tensor(self.t_feat, dtype=torch.float32)
            self.text_embedding = nn.Embedding.from_pretrained(t, freeze=False).to(self.device)
            self.text_trs = nn.Linear(t.size(1), self.embedding_dim).to(self.device)

        # ID-guided gates for content purification (MGCN-style)
        self.mm_gate_img = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2), nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim)
        ).to(self.device)
        self.mm_gate_txt = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2), nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim)
        ).to(self.device)

        self.mm_residual_alpha = config['mm_residual_alpha']
        self.mm_gate_temp = config['mm_gate_temp']
        self.mm_gate_cap = config['mm_gate_cap']

        # Build or load adjacency
        if (self.v_feat is not None) or (self.t_feat is not None):
            mm_cache = os.path.join(_CACHE_DIR, f"mm_adj_k{self.knn_k}_w{int(100 * self.mm_image_weight)}.pt")
            if os.path.exists(mm_cache):
                self.mm_adj = torch.load(mm_cache, map_location=self.device)
            else:
                image_adj = text_adj = None
                if self.v_feat is not None:
                    _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight)
                if self.t_feat is not None:
                    _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight)

                if (image_adj is not None) and (text_adj is not None):
                    self.mm_adj = (self.mm_image_weight * image_adj
                                   + (1.0 - self.mm_image_weight) * text_adj).coalesce()
                else:
                    self.mm_adj = (image_adj or text_adj)

                if self.mm_adj is not None:
                    torch.save(self.mm_adj.cpu(), mm_cache)
                    self.mm_adj = self.mm_adj.to(self.device)
        print(self.mm_adj)

    def _mm_purified_features(self):
        """
        ID-anchored content purification: gate is a function of projected content,
        and we anchor with item ID embeddings.
        """
        feats = []
        idw = self.item_embedding.weight  # (N,d)

        if hasattr(self, 'image_trs'):
            img_proj = self.image_trs(self.image_embedding.weight)  # (N,d)
            g_img = torch.sigmoid(self.mm_gate_img(img_proj))  # (N,d)
            feats.append(idw * g_img)

        if hasattr(self, 'text_trs'):
            txt_proj = self.text_trs(self.text_embedding.weight)  # (N,d)
            g_txt = torch.sigmoid(self.mm_gate_txt(txt_proj))  # (N,d)
            feats.append(idw * g_txt)

        if not feats:
            return None
        return sum(feats) / len(feats)

    def _mm_item_embeddings(self):
        """Apply 0–1 LightGCN hops on frozen MM adjacency to get item MM embeddings."""
        if (getattr(self, 'mm_adj', None) is None) or (self.n_mm_layers <= 0):
            return None
        x = self._mm_purified_features()
        if x is None:
            return None
        for _ in range(self.n_mm_layers):
            x = torch.sparse.mm(self.mm_adj, x)
        return x  # (N_items, d)

    def get_knn_adj_mat(self, mm_embeddings: torch.Tensor):
        # L2-normalize rows, cosine sim = dot product
        context_norm = mm_embeddings / (mm_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        sim = torch.mm(context_norm, context_norm.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim

        # Build sparse indices (row i → its kNN)
        indices0 = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((indices0.reshape(-1), knn_ind.reshape(-1)), dim=0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices: torch.Tensor, adj_size: torch.Size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size).coalesce()
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=-1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size).coalesce()

    @torch.no_grad()  # <<< EDIT  (no autograd for cache write)
    def _refresh_subset(self, item_vec, subseq_ids):
        if subseq_ids.numel() == 0:
            return
        itm = item_vec[self.sub_items_buf[subseq_ids]]  # (k,L,d)
        msk = self.sub_mask_buf[subseq_ids]
        self.subseq_cache.weight.data[subseq_ids] = self._seq_pool(itm, msk)  # <<< EDIT

    def _refresh_all(self, item_vec):
        seq_emb = self._seq_pool(item_vec[self.sub_items_buf], self.sub_mask_buf)
        with torch.no_grad():  # avoid leaf-grad clash
            self.subseq_cache.weight.data.copy_(seq_emb)  # <<< EDIT
        return seq_emb  # keep graph copy

    #################################################
    # 3.1) USER–ITEM GRAPH SETUP
    #################################################


    def get_norm_adj_mat(self):
        """
        Build the normalized user–item adjacency matrix (like FREEDOM).
        """
        import numpy as np
        import scipy.sparse as sp

        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(
            zip(inter_M.row, inter_M.col + self.n_users),
            [1] * inter_M.nnz
        ))
        data_dict.update(dict(zip(
            zip(inter_M_t.row + self.n_users, inter_M_t.col),
            [1] * inter_M_t.nnz
        )))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_edge_info(self):
        """
        Return (edge_indices, edge_values) for user–item adjacency dropout.
        """
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        """
        Normalize adjacency for partial dropout (user–item).
        """
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def pre_epoch_processing(self):
        """
        Dropout edges in user–item adjacency.
        """
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj

        else:
            degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout))
            degree_idx = torch.multinomial(self.edge_values, degree_len)
            keep_indices = self.edge_indices[:, degree_idx]
            keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
            all_values = torch.cat((keep_values, keep_values))
            keep_indices[1] += self.n_users
            all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), dim=1)
            self.masked_adj = torch.sparse.FloatTensor(
                all_indices, all_values, self.norm_adj.shape
            ).to(self.device)
        # with torch.no_grad():
        #     i_mm = self._mm_item_embeddings()
        #     base_items = i_mm if i_mm is not None else self.item_embedding.weight
        #     self._refresh_all(base_items)

    #################################################
    # 3.2) SUBSEQUENCE GRAPH BUILDING
    #################################################
    def get_subsequence_adj_mat(self):
        """
        Build or load a subsequence adjacency that includes:
          - Subsequence-Subsequence edges (via Levenshtein distance)
          - Subsequence-Item edges
        """
        # fetch the dictionary: subseq_map: {s_id: [item_ids]}
        # fetch n_items, dist_threshold
        # call build_subseq_graph_with_items(...) defined above
        ss_item_adj = build_subseq_graph_with_items(
            subseq_map=self.subseq_map,
            n_items=self.n_items,
            dist_threshold=self.dist_threshold
        )
        return ss_item_adj

    #################################################
    # 3.3) FORWARD PASS
    #################################################
    def forward(self, ui_adj,
                batch_users         = None,
                batch_pos_items     = None,
                batch_neg_items     = None,
                seq_items           = None,
                seq_mask            = None):
        """
        1) Use user–item adjacency to update user & item embeddings
        2) Use subsequence–item adjacency to update item & subsequence embeddings
        """
        # USER–ITEM PASS
        # unify user & item embeddings in one big matrix
        # shape: [n_users + n_items, embedding_dim]
        raw_item_embed = self.item_embedding.weight
        ui_ego = torch.cat((self.user_embedding.weight, raw_item_embed), dim=0)

        # n_ui_layers of GNN over user–item adjacency
        for _ in range(self.n_ui_layers):
            ui_ego = torch.sparse.mm(ui_adj, ui_ego)

        # split back out
        u_ui, i_ui = torch.split(
            ui_ego, [self.n_users, self.n_items], dim=0
        )

        s_id = s_mm = None
        i_mm = self._mm_item_embeddings()  # (N_items, d) or None

        if (seq_items is not None) and (self.refresh_mode == "lazy"):
            # 1) (B, L, d) fresh sequence vectors (on the graph)
            s_id = self.att_id(i_ui[seq_items], seq_mask)
            fresh_seq = s_id

            s_mm = self.att_mm(i_mm(seq_items), seq_mask)

            # 2) start from cache, splice‑in the B fresh rows
            seq_input = self.subseq_cache.weight.clone()  # (N_sub, d)
            seq_input[batch_users] = fresh_seq

            # 3) update cache out‑of‑graph
            with torch.no_grad():
                self.subseq_cache.weight.data[batch_users] = fresh_seq.detach()

        else:  # evaluation or 'full' mode
            seq_input = self.subseq_cache.weight

        si_ego = torch.cat((raw_item_embed, seq_input), dim=0)
        # shape => [n_items + n_subseq, embedding_dim]

        # n_si_layers of GNN over subsequence–item adjacency
        for _ in range(self.n_ss_layers):
            si_ego = torch.sparse.mm(self.si_adj, si_ego)

        # split back out
        new_item_emb, new_seq_emb = torch.split(
            si_ego, [self.n_items, self.n_subseq], dim=0
        )

        s_mm = s_mm
        s_id = new_seq_emb[batch_users]

        if self.refresh_mode == "lazy":
            with torch.no_grad():
                self.subseq_cache.weight.data.copy_(new_seq_emb.detach())

        return u_ui, i_ui, s_id, s_mm, i_mm

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return mf_loss

    def info_nce(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
        """
        a, b : (B, d) ℓ2-normalised embeddings from two views.
        returns scalar InfoNCE loss.
        """
        sim = torch.mm(a, b.t()) / tau  # (B, B) cosine similarities
        sim_exp = torch.exp(sim)
        pos = torch.diag(sim_exp)  # positives: same row/col
        denom = sim_exp.sum(dim=1)  # all pairs
        return -(torch.log(pos / denom)).mean()  # alignment + uniformity

    def calculate_loss(self, interaction):
        batch_tensor, seq_items, seq_mask = interaction
        users = batch_tensor[0].to(self.device)
        pos_items = batch_tensor[1].to(self.device)
        neg_items = batch_tensor[2].to(self.device)

        u_ui, i_ui, s_id, s_mm, i_mm = self.forward(self.masked_adj,
                                                    users, pos_items, neg_items,
                                                    seq_items, seq_mask)

        # 1) BPR on collaborative (UI) view
        # BPR with a tiny, gated MM residual into item embeddings (only if i_mm exists and alpha>0)
        if (i_mm is not None) and (getattr(self, 'mm_residual_alpha', 0.0) > 0.0):
            i_mm_n = F.normalize(i_mm, dim=1)
            i_ui_n = F.normalize(i_ui, dim=1)

            # gates per selected items (agreement between CF and MM)
            cos_pos = (i_mm_n[pos_items] * i_ui_n[pos_items]).sum(dim=1, keepdim=True)  # (B,1)
            gate_pos = torch.sigmoid(self.mm_gate_temp * cos_pos).clamp(max=self.mm_gate_cap)
            i_pos = i_ui[pos_items] + self.mm_residual_alpha * gate_pos * i_mm_n[pos_items]

            cos_neg = (i_mm_n[neg_items] * i_ui_n[neg_items]).sum(dim=1, keepdim=True)  # (B,1)
            gate_neg = torch.sigmoid(self.mm_gate_temp * cos_neg).clamp(max=self.mm_gate_cap)
            i_neg = i_ui[neg_items] + self.mm_residual_alpha * gate_neg * i_mm_n[neg_items]
        else:
            i_pos, i_neg = i_ui[pos_items], i_ui[neg_items]

        mf_loss = self.bpr_loss(u_ui[users], i_pos, i_neg)

        # 2) Seq-view InfoNCE (s_id vs s_mm)
        loss_seqcl = 0.0
        if (s_id is not None) and (s_mm is not None):
            z_sid = F.normalize(s_id, dim=1)
            z_smm = F.normalize(s_mm, dim=1)
            loss_seqcl = self.info_nce(z_sid, z_smm, tau=self.tau)

        # 3) Item ID ↔ MM InfoNCE (use batch positives)
        loss_itemcl = 0.0
        if i_mm is not None:
            z_iid = F.normalize(i_ui[pos_items], dim=1)
            z_imm = F.normalize(i_mm[pos_items], dim=1)
            loss_itemcl = self.info_nce(z_iid, z_imm, tau=self.tau)

        # 4) (Optional) user ↔ sequence InfoNCE (small weight)
        loss_userseq = 0.0
        if (s_id is not None) and (getattr(self, 'lambda_userseqcl', 0.0) > 0):
            loss_userseq = self.info_nce(F.normalize(u_ui[users], dim=1),
                                         F.normalize(s_id, dim=1), tau=self.tau)

        wandb.log({'mf_loss': mf_loss,
                   'loss_seqcl': loss_seqcl,
                   'loss_itemcl': loss_itemcl,
                   'loss_userseq': loss_userseq})

        return (mf_loss
                + getattr(self, 'lambda_seqcl', 0.0) * loss_seqcl
                + getattr(self, 'lambda_itemcl', 0.0) * loss_itemcl
                + getattr(self, 'lambda_userseqcl', 0.0) * loss_userseq)

    def full_sort_predict(self, interaction):
        """
        For top-K item ranking.
        Compute user->all-items dot products.
        """
        user = interaction[0]
        restore_user_e, i_ui, _, _, i_mm = self.forward(self.norm_adj)
        u_embedding = restore_user_e[user]
        item_e = i_ui

        if (i_mm is not None) and (getattr(self, 'mm_residual_alpha', 0.0) > 0.0):
            i_mm_n = F.normalize(i_mm, dim=1)
            i_ui_n = F.normalize(i_ui, dim=1)
            cos = (i_mm_n * i_ui_n).sum(dim=1, keepdim=True)  # (N_items,1)
            gate = torch.sigmoid(self.mm_gate_temp * cos).clamp(max=self.mm_gate_cap)
            item_e = i_ui + self.mm_residual_alpha * gate * i_mm_n

        scores = torch.matmul(u_embedding, item_e.transpose(0, 1))
        return scores

class AdditiveAttentionPool(nn.Module):
    """
    Single-head additive attention pooling.
    Input:  item_emb   (B, L, d)   – item embeddings, d = 64
            mask       (B, L)      – 1 for valid items, 0 for padding
    Output: seq_emb    (B, d)      – pooled subsequence embedding
    """
    def __init__(self, d_model: int = 64, d_hidden: int = 32, max_len: int = 50):
        super().__init__()
        # W_a : (d_hidden, d_model) ; v_a : (d_hidden)
        self.W_a = nn.Linear(d_model, d_hidden, bias=True)
        self.v_a = nn.Linear(d_hidden, 1, bias=False)

        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, item_emb: torch.Tensor, mask: torch.Tensor):
        # item_emb: (B, L, d)
        # mask:     (B, L)
        B, L, d = item_emb.size()
        assert L <= self.pos_emb.num_embeddings, "sequence longer than max_len"

        # ---- add positional embeddings ------------------------------
        pe = self.pos_emb.weight[:L].unsqueeze(0)  # (1,L,d)
        x = item_emb + pe  # (B,L,d)

        # ---- attention ------------------------------------------------
        e = self.v_a(torch.tanh(self.W_a(x))).squeeze(-1)  # (B,L)
        e = e.masked_fill(mask == 0, -1e9)
        a = F.softmax(e, dim=1).unsqueeze(-1)  # (B,L,1)
        seq_emb = torch.sum(a * x, dim=1)  # (B,d)

        return seq_emb          # Already size 64 if d_model=64

class SinusoidalPool(nn.Module):
    """Parameter–free subsequence encoder (mean‑pool + fixed PE)."""
    def __init__(self, d_model: int = 64, max_len: int = 50):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)                # (L,1)
        two_i = torch.arange(0, d_model, 2).float()                     # even dims
        div  = torch.exp(-torch.log(torch.tensor(10000.0)) * two_i / d_model)
        pe   = torch.zeros(max_len, d_model)                            # (L,d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)                                  # (L,d)

    def forward(self, item_emb: torch.Tensor, mask: torch.Tensor):      # (B,L,d)
        pe   = self.pe[: item_emb.size(1)].unsqueeze(0)                 # (1,L,d)
        x    = (item_emb + pe) * mask.unsqueeze(-1)                     # zero pads
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        return x.sum(1) / denom                                         # (B,d)