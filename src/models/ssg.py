import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

    print("correct file is running")
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
class SSG(GeneralRecommender):
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
        super(SSG, self).__init__(config, dataset)

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
        self.uigraph_weight = 1.0  # weight factor for BPR loss


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
            self.att_pool = AdditiveAttentionPool(64, 32).to(self.device)
        elif self.pooling_type == "fastatt":
            self.att_pool = FastDotAttentionPool(64)
        elif self.pooling_type == "sinus":
            self.att_pool = SinusoidalPool(64, max_len=self.max_len)
        elif self.pooling_type == "flashatt":
            self.att_pool = FlashAttentionPool(
                embed_dim=self.embedding_dim,
                n_heads=4)

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

        # # Create trainable embedding for each subsequence
        # self.subseq_embedding = nn.Embedding(self.n_subseq, self.embedding_dim).to(self.device)
        # nn.init.xavier_uniform_(self.subseq_embedding.weight)
        # self.subseq_embedding.weight.requires_grad_(False)

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

        # # turn every list into a torch tensor for fast gather
        # self.item2subseq = [torch.tensor(lst, dtype=torch.long)
        #                     if lst else None  # None = no subseq uses item
        #                     for lst in inv]

        # Build subsequence adjacency = SS + SI
        if self.beta ==0 and self.n_ss_layers==0:
            self.si_adj = None
        else:
            self.si_adj = self.get_subsequence_adj_mat().coalesce().to(self.device)

        self.refresh_mode = "lazy"
        # with torch.no_grad():  # one warm start
        #     self.subseq_cache.weight.data.copy_(
        #         self.att_pool(
        #             self.item_embedding.weight[self.sub_items_buf],
        #             self.sub_mask_buf
        #         )
        #     )
        # self.refresh_mode = "full"

    @torch.no_grad()  # <<< EDIT  (no autograd for cache write)
    def _refresh_subset(self, item_vec, subseq_ids):
        if subseq_ids.numel() == 0:
            return
        itm = item_vec[self.sub_items_buf[subseq_ids]]  # (k,L,d)
        msk = self.sub_mask_buf[subseq_ids]
        self.subseq_cache.weight.data[subseq_ids] = self.att_pool(itm, msk)  # <<< EDIT

    def _refresh_all(self, item_vec):
        seq_emb = self.att_pool(item_vec[self.sub_items_buf],  # (Nsub,L,d)
                                self.sub_mask_buf)  # (Nsub,d)
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
        with torch.no_grad():
            self._refresh_all(self.item_embedding.weight)

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
        # ----------------------------
        # A) USER–ITEM PASS
        # ----------------------------
        # Step A1: unify user & item embeddings in one big matrix
        # shape: [n_users + n_items, embedding_dim]
        raw_item_embed = self.item_embedding.weight
        ui_ego = torch.cat((self.user_embedding.weight, raw_item_embed), dim=0)

        # Step A2: n_ui_layers of GNN over user–item adjacency
        for _ in range(self.n_ui_layers):
            ui_ego = torch.sparse.mm(ui_adj, ui_ego)

        # Step A3: split back out
        updated_user_embed, updated_item_embed = torch.split(
            ui_ego, [self.n_users, self.n_items], dim=0
        )

        # ----------------------------
        # B) SUBSEQUENCE–ITEM PASS
        # ----------------------------
        # We'll now unify 'updated_item_embed' with 'subseq_embedding.weight'
        # in a single big matrix of shape [n_items + n_subseq, embedding_dim].

        #seq_emb = self.compute_all_subseq_emb(updated_item_embed)

        # ---------------- B) build / fetch subsequence vectors ----------------
        # if self.refresh_mode == "full":
        #     # unchanged – everything in-graph
        #     seq_input = self._refresh_all(updated_item_embed)
        #
        # elif self.refresh_mode == "lazy" and batch_pos_items is not None:
        #     # 1) which items appear in this mini-batch ?
        #     touched_items = torch.unique(
        #         torch.cat((batch_pos_items, batch_neg_items)))  # (m,)
        #
        #     touched_subseq = torch.unique(torch.cat([
        #         self.item2subseq_flat[self.item2subseq_ptr[t]: self.item2subseq_ptr[t + 1]]
        #         for t in touched_items
        #     ])).to(self.device)
        #
        #     if touched_subseq.numel() > 0:
        #         sub_itm = updated_item_embed[self.sub_items_buf[touched_subseq]]
        #         sub_mask = self.sub_mask_buf[touched_subseq]
        #         new_emb = self.att_pool(sub_itm, sub_mask)
        #
        #         # ❶ make *view* that is not cloned
        #         seq_input = self.subseq_cache.detach()
        #         # ❷ ensure only the k rows participate in autograd
        #         seq_slice = seq_input[touched_subseq].clone()  # small clone
        #         seq_slice.copy_(new_emb)  # grads flow
        #         seq_input = seq_input.clone()  # shallow clone header
        #         seq_input[touched_subseq] = seq_slice
        #
        #         # ❸ background cache update
        #         with torch.no_grad():
        #             self.subseq_cache[touched_subseq] = new_emb
        #     else:
        #         seq_input = self.subseq_cache.detach()
        #
        # else:  # evaluation or no item information
        #     seq_input = self.subseq_cache.detach()

        if seq_items is not None and self.refresh_mode == "lazy":
            # 1) (B, L, d) fresh sequence vectors (on the graph)
            fresh_seq = self.att_pool(
                self.item_embedding(seq_items),  # item_emb
                seq_mask  # pad mask
            )  # (B, d)

            # 2) start from cache, splice‑in the B fresh rows
            seq_input = self.subseq_cache.weight.clone()  # (N_sub, d)
            seq_input[batch_users] = fresh_seq

            # 3) update cache out‑of‑graph
            with torch.no_grad():
                self.subseq_cache.weight.data[batch_users] = fresh_seq.detach()

        else:  # evaluation or 'full' mode
            seq_input = self.subseq_cache.weight

        # seq_input = self.att_pool(
        #                 raw_item_embed[self.sub_items_buf],  # (N_sub,L,d)
        #                 self.sub_mask_buf  # (N_sub,L)
        #     )

        si_ego = torch.cat((seq_input, raw_item_embed), dim=0)
        # si_ego = torch.cat((raw_item_embed, seq_input), dim=0)
        # shape => [n_items + n_subseq, embedding_dim]

        # Step B2: n_si_layers of GNN over subsequence–item adjacency
        for _ in range(self.n_ss_layers):
            si_ego = torch.sparse.mm(self.si_adj, si_ego)

        # Step B3: split back out
        new_subseq_embed, new_item_embed = torch.split(
            si_ego, [self.n_subseq, self.n_items], dim=0
        )

        # Now 'new_item_embed' has been updated from both user adjacency (part A)
        # and subsequence adjacency (part B).
        # 'new_subseq_embed' is our updated subsequence embedding.

        final_item = updated_item_embed + self.beta * new_item_embed

        if self.refresh_mode == "lazy":
            with torch.no_grad():
                self.subseq_cache.weight.data.copy_(new_subseq_embed.detach())

        if self.use_user_seq_fusion and self.alpha != 0.0:
            if self.fusion_norm:
                u = F.normalize(updated_user_embed, dim=1)
                su = F.normalize(new_subseq_embed, dim=1)
                final_item = F.normalize(final_item, dim=1)
            else:
                u, su = updated_user_embed, new_subseq_embed
            combined_user = u + self.alpha * su
        else:
            combined_user = updated_user_embed

        return combined_user, final_item, new_subseq_embed

    #################################################
    # 3.4) TRAINING OBJECTIVES
    #################################################
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return mf_loss

    def calculate_loss(self, interaction):
        """
        Typical BPR loss on user–item pairs.
        Subsequence embeddings might be included in a future extension
        (e.g., a subsequence–item alignment loss).
        """
        batch_tensor, seq_items, seq_mask = interaction
        users = batch_tensor[0].to(self.device)
        pos_items = batch_tensor[1].to(self.device)
        neg_items = batch_tensor[2].to(self.device)

        ua_embeddings, ia_embeddings, sa_embeddings = self.forward(self.masked_adj, users, pos_items, neg_items, seq_items, seq_mask)
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        # Potentially add extra losses for subsequence alignment if desired
        return self.uigraph_weight * batch_mf_loss

    def full_sort_predict(self, interaction):
        """
        For top-K item ranking.
        Compute user->all-items dot products.
        """
        user = interaction[0]
        restore_user_e, restore_item_e, _ = self.forward(self.norm_adj)
        u_embedding = restore_user_e[user]
        scores = torch.matmul(u_embedding, restore_item_e.transpose(0, 1))
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

class FastDotAttentionPool(nn.Module):
    """
    Single‑head *dot‑product* attention pooling.
    – only one matrix–vector multiply instead of two matrix–matrix multiplies
    – same output shape (B, d)
    """
    def __init__(self, d_model: int = 64, max_len: int = 50):
        super().__init__()
        # learnable query q  (d,)
        self.q = nn.Parameter(torch.randn(d_model))
        nn.init.xavier_uniform_(self.q.unsqueeze(0))

        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, item_emb: torch.Tensor, mask: torch.Tensor):
        # item_emb : (B, L, d)
        # mask     : (B, L)   — 1 = keep, 0 = pad
        # ------------------------- score --------------------------
        #   score_i = qᵀ · h_i     ->  einsum is fastest for this shape
        B, L, d = item_emb.size()
        assert L <= self.pos_emb.num_embeddings, "sequence longer than max_len"

        x = item_emb + self.pos_emb.weight[:L].unsqueeze(0)  # (B,L,d)

        scores = torch.einsum('bld,d->bl', x, self.q)        # (B,L)

        # --------------------- masked soft‑max --------------------
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = F.softmax(scores, dim=1).unsqueeze(-1)             # (B,L,1)

        # ---------------------- weighted sum ----------------------
        return torch.sum(attn * item_emb, dim=1)                    # (B,d)

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

class FlashAttentionPool(nn.Module):
    """
    One‑token query → {K,V}=sequence   ➜  pooled vector (B, D)
    Uses PyTorch SDPA which calls FlashAttention‑2 on CUDA.
    """
    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads

        # learnable global query (same idea as additive‑attention’s vector 'q')
        q0 = torch.randn(1, n_heads, 1, self.head_dim) / self.head_dim**0.5
        self.q = nn.Parameter(q0)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        """
        x        : (B, L, D) float16/32
        pad_mask : (B, L)   bool, True for *real* tokens (same polarity as your current code)
        returns  : (B, D)
        """
        B, L, D = x.shape
        # (B, L, D) ➜ (B, nH, L, dH)
        kv = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # expand the 1‑token query
        q = self.q.expand(B, -1, -1, -1)                # (B, nH, 1, dH)

        # PyTorch SDPA expects attn_mask == True where positions are *masked*
        attn_mask = None
        if pad_mask is not None:
            attn_mask = (~pad_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        # FlashAttention kernel is invoked under the hood
        out = F.scaled_dot_product_attention(
            q, kv, kv,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )                                               # (B, nH, 1, dH)

        return out.squeeze(2).reshape(B, D)             # (B, D)
