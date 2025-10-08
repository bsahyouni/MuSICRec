import torch
from recbole.data.interaction import Interaction
import tqdm
import os, io, pandas as pd

def rebuild_rb_inter(data_root: str, dataset: str, sort_for_repro: bool = False):
    """Create <dataset>_rb/<dataset>_rb.inter from <dataset>/<dataset>.inter,
    fixing header types and tabs so RecBole can load it."""
    src_dir = os.path.join(data_root, dataset)
    src = os.path.join(src_dir, f"{dataset}.inter")
    assert os.path.isfile(src), f"[RecBole] Missing source atomic file: {src}"

    rb_dataset = f"{dataset}_rb"
    dst_dir = os.path.join(data_root, rb_dataset)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{rb_dataset}.inter")

    # --- read & sanitize header ---
    with open(src, "rb") as f:
        raw = f.read()
    # handle potential UTF-8 BOM
    text = raw.decode("utf-8-sig")

    first_nl = text.find("\n")
    assert first_nl != -1, f"[RecBole] {src} has no newline / data rows?"
    header = text[:first_nl].rstrip("\r\n")
    body   = text[first_nl+1:]

    # split header on tabs, drop empties
    tokens = [t for t in header.split("\t") if t]
    # add missing :type where needed (heuristic for common fields)
    fixed = []
    for t in tokens:
        if ":" in t:
            name, typ = t.split(":", 1)
            name = name.strip()
            typ  = typ.strip()
            if not name or not typ:   # guard odd cases like "userID:"
                # infer again below
                t = name
            else:
                fixed.append(f"{name}:{typ}")
                continue
        # infer type for untyped column name
        name_l = t.strip()
        low = name_l.lower()
        if "user" in low:
            fixed.append(f"{name_l}:token")
        elif "item" in low:
            fixed.append(f"{name_l}:token")
        elif "time" in low or "ts" in low:
            fixed.append(f"{name_l}:float")
        elif low in {"label", "rating", "score"}:
            fixed.append(f"{name_l}:float")
        else:
            # default to token if unknown
            fixed.append(f"{name_l}:token")

    # rebuild a clean, tab-separated header
    clean_header = "\t".join(fixed)

    # fast path: if no sorting requested, just rewrite with clean header + body
    if not sort_for_repro:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(clean_header + "\n")
            # normalize body line endings and ensure tab separators
            f.write(body.replace("\r\n", "\n").replace("\r", "\n"))
        return rb_dataset, fixed

    # optional: sort for reproducibility (not required for RecBole splitting)
    col_names = [c.split(":", 1)[0] for c in fixed]
    df = pd.read_csv(io.StringIO(body), sep="\t", header=None, names=col_names, dtype=str)
    # pick actual field names
    def pick(substr, fallback):
        for c in col_names:
            if substr in c.lower():
                return c
        return fallback
    u = pick("user", "userID")
    i = pick("item", "itemID")
    t = pick("time", "timestamp")
    # coerce types
    for c in (u, i):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if t in df:
        df[t] = pd.to_numeric(df[t], errors="coerce")  # float OK
    df = df.sort_values([c for c in (u, t, i) if c in df], kind="mergesort")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(clean_header + "\n")
    df.to_csv(dst, sep="\t", header=False, index=False, mode="a")
    return rb_dataset, fixed

def _strict_twostep_eval_general(model, device, item_num,
                                 users_eval,       # List[int] INTERNAL user ids
                                 train_pos,        # Dict[int, Set[int]] items in TRAIN (strict up to t_{n-2})
                                 targets,          # Dict[int, int] target per user (val or test)
                                 stage="valid",    # "valid" or "test"
                                 val_targets=None,
                                 topk_list=(5, 10, 20, 50),
                                 batch_users=64):
    """
    Strict two-step evaluator for GENERAL RecBole models (e.g., LightGCN, SGL, BPR).
    Micro-batched, no_grad, optional AMP to avoid [B, n_items] OOM during full_sort_predict.
    """
    import torch
    from recbole.data.interaction import Interaction

    if not users_eval:
        return {f'{m}@{k}': 0.0 for m in ('recall','ndcg','precision','map') for k in topk_list}

    # accumulators
    metric_sums = {f"{m}@{k}": 0.0 for m in ("recall","precision","map","ndcg") for k in topk_list}
    total = 0
    # process users in chunks
    for s in range(0, len(users_eval), batch_users):
        u_batch = users_eval[s:s+batch_users]
        if not u_batch:
            continue

        u_tensor = torch.tensor(u_batch, dtype=torch.long, device=device)
        inter = Interaction({ model.USER_ID: u_tensor })

        with torch.no_grad():
            # full_sort_predict returns [b, n_items]
            scores = model.full_sort_predict(inter)

        # PAD at index 0 in RecBole => mask it
        if scores.size(1) > 0:
            scores[:, 0] = float("-inf")

        # mask seen items (TRAIN) and, at test, also mask validation item; keep target visible
        for b, uid in enumerate(u_batch):
            hist = set(train_pos.get(uid, ()))
            if stage == "test" and val_targets is not None:
                vtar = int(val_targets.get(uid, -1))
                ttar = int(targets[uid])
                if vtar > 0 and vtar != ttar:
                    hist.add(vtar)
            tgt = int(targets[uid])
            if tgt in hist:
                hist.remove(tgt)
            if hist:
                idx = torch.tensor(list(hist), dtype=torch.long, device=scores.device)
                scores[b, idx] = float("-inf")

        # compute metrics on this chunk
        tgt_idx = torch.tensor([int(targets[u]) for u in u_batch], dtype=torch.long, device=scores.device)
        bsz = scores.size(0)
        for k in topk_list:
            topk_idx = torch.topk(scores, k, dim=1).indices
            hit = (topk_idx == tgt_idx.view(-1,1)).any(dim=1).float()         # [b]
            metric_sums[f"recall@{k}"]    += hit.mean().item() * bsz          # recall == hit for single target
            metric_sums[f"precision@{k}"] += (hit / k).mean().item() * bsz

            pos = (topk_idx == tgt_idx.view(-1,1)).float().argmax(dim=1)
            found = hit.bool()
            ap  = torch.zeros_like(hit);  ap[found]  = 1.0 / (pos[found].float() + 1.0)
            dcg = torch.zeros_like(hit); dcg[found] = 1.0 / torch.log2(pos[found].float() + 2.0)
            metric_sums[f"map@{k}"]  += ap.mean().item()  * bsz
            metric_sums[f"ndcg@{k}"] += dcg.mean().item() * bsz

        # free up scores ASAP
        del scores
        total += bsz

    metrics = {name: (val / max(total, 1)) for name, val in metric_sums.items()}
    return metrics


def _strict_twostep_eval_context(model, device, item_num,
                                 users_eval,
                                 train_pos,
                                 targets,
                                 ctx_tensors,          # Dict[field -> torch.Tensor[B]]  (val or test context)
                                 stage="valid",
                                 val_targets=None,
                                 topk_list=(5, 10, 20, 50)):
    """Strict two-step evaluator for CONTEXT-AWARE models.
       Supplies per-user context features (taken from the corresponding val/test interaction)."""

    if not users_eval:
        return {f'{m}@{k}': 0.0 for m in ('recall','ndcg','precision','map') for k in topk_list}

    payload = { model.USER_ID: torch.tensor(users_eval, dtype=torch.long, device=device) }
    # attach context features
    for f, t in ctx_tensors.items():
        payload[f] = t.to(device)
    inter = Interaction(payload)

    with torch.no_grad():
        scores = model.full_sort_predict(inter).view(len(users_eval), -1).cpu()
    if scores.size(1) > item_num:
        scores = scores[:, :item_num]
    scores[:, 0] = float("-inf")  # PAD

    for b, uid in enumerate(users_eval):
        hist = set(train_pos.get(uid, ()))
        if stage == "test" and val_targets is not None:
            vtar = int(val_targets.get(uid, -1))
            if vtar > 0 and vtar != int(targets[uid]):
                hist.add(vtar)
        tgt = int(targets[uid])
        hist.discard(tgt)
        if hist:
            scores[b, list(hist)] = float("-inf")

    tgt_idx = torch.tensor([int(targets[u]) for u in users_eval], dtype=torch.long)
    metrics = {}
    for k in topk_list:
        topk_idx = torch.topk(scores, k, dim=1).indices
        hit = (topk_idx == tgt_idx.view(-1,1)).any(dim=1).float()
        metrics[f"recall@{k}"]   = hit.mean().item()
        metrics[f"precision@{k}"] = (hit / k).mean().item()
        pos = (topk_idx == tgt_idx.view(-1,1)).float().argmax(dim=1)
        found = hit.bool()
        ap  = torch.zeros_like(hit); ap[found]  = 1.0 / (pos[found].float() + 1.0)
        dcg = torch.zeros_like(hit); dcg[found] = 1.0 / torch.log2(pos[found].float() + 2.0)
        metrics[f"map@{k}"]  = ap.mean().item()
        metrics[f"ndcg@{k}"] = dcg.mean().item()
    return metrics

def _pad_sequences(prefixes, pad_id=0, max_len=50):
    # prefixes: List[List[int]]  (items must be at the front; PADs at the end)
    arr = []
    lens = []
    for p in prefixes:
        # keep at most max_len tokens, from the FRONT (RecBole already provides L <= MAX)
        p = p[:max_len]
        L = len(p)
        lens.append(L)
        # RIGHT-PAD with PAD=0 → items first, then PADs (RecBole convention)
        arr.append(p + [pad_id] * (max_len - L))
    return torch.tensor(arr, dtype=torch.long), torch.tensor(lens, dtype=torch.long)

def _strict_twostep_eval_sasfamily(model, device, item_num,
                                   user2seq, train_pos,  # dict: uid -> list[int], set[int]
                                   targets,              # dict: uid -> int (val or test)
                                   max_len=50, pad_id=0,
                                   topk_list=(5,10,20,50),
                                   stage='valid',
                                   val_targets=None,
                                   batch_users=256):
    """
    Strict two-step eval for SAS/Transformer family with micro-batching to avoid OOM.
    """
    model.eval()

    # ---- pack inputs from user2seq ----
    uids, prefixes, tars = [], [], []
    for uid, seq in user2seq.items():
        if len(seq) == 0:
            continue
        uids.append(uid)
        prefixes.append(seq)
        tars.append(targets[uid])
    if not uids:
        return {f'{m}@{k}': 0.0 for m in ('recall','ndcg','precision','map') for k in (5,10,20,50)}

    item_id_list, item_length = _pad_sequences(prefixes, pad_id, max_len)

    # sanity on padding & ids
    idx = torch.arange(item_length.size(0))
    last_tokens = item_id_list[idx, (item_length - 1).clamp_min(0)]
    zero_last = (last_tokens == pad_id).sum().item()
    print(f"[dbg] last_tokens_zero={zero_last}/{item_id_list.size(0)} (should be 0)")

    non_pad = item_id_list[item_id_list > pad_id]
    if non_pad.numel() > 0:
        mn = int(non_pad.min().item())
        mx = int(non_pad.max().item())
        assert mn >= 1, f"Found non-PAD id {mn} < 1 (looks 0-based). Reindex to 1-based."
        assert mx < item_num, f"id {mx} >= item_num({item_num}). Check internal-id mapping."

    # ---- micro-batched forward to save memory ----
    B = item_id_list.size(0)

    metric_sums = {f"{m}@{k}": 0.0 for m in ("recall","precision","map","ndcg") for k in topk_list}
    total = 0

    for s in range(0, B, batch_users):
        e = min(s + batch_users, B)
        item_id_list_b = item_id_list[s:e].to(device).long()
        item_length_b  = item_length[s:e].to(device).long()
        uids_b         = uids[s:e]
        tgt_idx_b      = torch.tensor([int(t) for t in tars[s:e]], dtype=torch.long, device=device)
        bsz = item_id_list_b.size(0)

        inter = Interaction({
            model.ITEM_SEQ: item_id_list_b,       # [b, L]
            model.ITEM_SEQ_LEN: item_length_b,    # [b]
        })

        with torch.no_grad():
            # full_sort_predict returns [b, n_items]
            scores = model.full_sort_predict(inter)  # [b, item_num]
        scores[:, pad_id] = float("-inf")

        # mask history (+ validation item at test), but keep the ground-truth target visible
        for b in range(bsz):
            hist = set(item_id_list_b[b].tolist())
            if pad_id in hist:
                hist.remove(pad_id)
            if stage == "test" and val_targets is not None:
                vtar = int(val_targets.get(uids_b[b], -1))
                ttar = int(tars[s + b])
                if vtar > 0 and vtar != ttar:
                    hist.add(vtar)
            tgt = int(tars[s + b])
            if tgt in hist:
                hist.remove(tgt)
            if hist:
                hist_idx = torch.tensor(list(hist), dtype=torch.long, device=scores.device)
                scores[b, hist_idx] = float("-inf")

        # chunk metrics
        for k in topk_list:
            topk_idx = torch.topk(scores, k, dim=1).indices           # [b, k]
            hit = (topk_idx == tgt_idx_b.view(-1,1)).any(dim=1).float()
            metric_sums[f"recall@{k}"]    += hit.mean().item() * bsz   # recall==hit for single target
            metric_sums[f"precision@{k}"] += (hit / k).mean().item() * bsz

            pos = (topk_idx == tgt_idx_b.view(-1,1)).float().argmax(dim=1)
            found = hit.bool()
            ap  = torch.zeros_like(hit);  ap[found]  = 1.0 / (pos[found].float() + 1.0)
            dcg = torch.zeros_like(hit); dcg[found] = 1.0 / torch.log2(pos[found].float() + 2.0)
            metric_sums[f"map@{k}"]  += ap.mean().item()  * bsz
            metric_sums[f"ndcg@{k}"] += dcg.mean().item() * bsz

        total += bsz

    # finalize
    metrics = {name: (val / max(total,1)) for name, val in metric_sums.items()}
    return metrics

# def _strict_twostep_eval_sasfamily(model, device, item_num,
#                                    user2seq, train_pos,  # dict: uid -> list[int], set[int]
#                                    targets,              # dict: uid -> int (val or test)
#                                    max_len=50, pad_id=0,
#                                    topk_list=(5,10,20,50),
#                                    stage='valid',
#                                    val_targets=None):
#     """
#     Build Interaction for each user's prefix up to t_{n-2} and predict full-catalog scores.
#     Mask PAD + prefix-history + (optionally) nothing else (train-only masking already handled by using strict prefixes).
#     """
#     model.eval()
#     uids = []
#     prefixes = []
#     tars = []
#     for uid, seq in user2seq.items():
#         # user2seq already holds the strict history up to t_{n-2} (train split)
#         if len(seq) == 0:
#             continue
#         uids.append(uid)
#         prefixes.append(seq)
#         tars.append(targets[uid])
#     if not uids:
#         return {f'{m}@{k}': 0.0 for m in ('recall','ndcg','precision','map') for k in (5,10,20,50)}
#
#     item_id_list, item_length = _pad_sequences(prefixes, pad_id, max_len)
#
#     # The last real token is at index item_length-1; it must be > 0 (not PAD).
#     idx = torch.arange(item_length.size(0))
#     last_tokens = item_id_list[idx, (item_length - 1).clamp_min(0)]
#     zero_last = (last_tokens == pad_id).sum().item()
#     print(f"[dbg] last_tokens_zero={zero_last}/{item_id_list.size(0)} (should be 0)")
#
#     print(f"[dbg] stage={stage} B={item_id_list.size(0)} L={item_id_list.size(1)} "
#           f"minID(non-PAD)={int(item_id_list[item_id_list > pad_id].min().item()) if (item_id_list > pad_id).any() else None} "
#           f"maxID={int(item_id_list[item_id_list > pad_id].max().item()) if (item_id_list > pad_id).any() else None} "
#           f"item_num={item_num}")
#     # Build RecBole Interaction expected by SASRec/FEARec full_sort_predict
#     # Field names used by sequential models in RecBole:
#     seq_field = model.ITEM_SEQ
#     len_field = model.ITEM_SEQ_LEN
#
#     inter = Interaction({
#         model.ITEM_SEQ: item_id_list.to(device).long(),  # shape [B, L]
#         model.ITEM_SEQ_LEN: item_length.to(device).long(),  # shape [B]
#     })
#
#     non_pad = item_id_list[item_id_list > pad_id]
#     if non_pad.numel() > 0:
#         mn = int(non_pad.min().item())
#         mx = int(non_pad.max().item())
#         assert mn >= 1, f"Found non-PAD id {mn} < 1 (looks 0-based). Reindex to 1-based."
#         assert mx < item_num, f"id {mx} >= item_num({item_num}). Check internal-id mapping."
#
#     with torch.no_grad():
#         scores = model.full_sort_predict(inter).cpu()  # [B, item_num]
#     B = item_id_list.size(0)  # define B before using it below
#
#     # mask PAD + history in prefix
#     scores[:, pad_id] = float('-inf')
#
#     # mask history EXCEPT the ground-truth target
#     # (simple, safe loop; vectorize later if needed)
#     # for b in range(B):
#     #     hist = set(item_id_list[b].tolist())
#     #     hist.discard(pad_id)
#     #     hist.discard(int(tars[b]))  # keep the target unmasked
#     #     if hist:
#     #         scores[b, list(hist)] = float("-inf")
#
#     for b in range(B):
#         hist = set(item_id_list[b].tolist())
#         hist.discard(pad_id)
#         # ++ also mask the validation item in TEST (two-step strictness)
#         if stage == "test" and val_targets is not None:
#             vtar = int(val_targets.get(uids[b], -1))
#             if vtar > 0 and vtar != int(tars[b]):  # valid id and not the test target
#                 hist.add(vtar)
#         hist.discard(int(tars[b]))  # keep the ground-truth target visible
#         if hist:
#             scores[b, list(hist)] = float("-inf")
#
#     # compute top-k metrics against each user's target
#     # Build ground-truth indices
#     tgt_idx = torch.tensor([t for t in tars], dtype=torch.long)  # already token ids
#     # rank
#     metrics = {f'{m}@{k}': 0.0 for m in ('recall','ndcg','precision','map') for k in (5,10,20,50)}
#     B, V = scores.shape
#     # targets must be valid ids
#     assert (tgt_idx >= 0).all() and (tgt_idx < V).all(), \
#         f"Targets out of range (min={int(tgt_idx.min())}, max={int(tgt_idx.max())}, V={V})"
#     # warn if any target got masked to -inf
#     masked_cnt = torch.isneginf(scores[torch.arange(B), tgt_idx]).sum().item()
#     print(f"[dbg] targets masked = {masked_cnt}/{scores.size(0)} (0 expected)")
#     if masked_cnt:
#         tqdm.write(f"[strict] WARNING: {masked_cnt}/{B} targets are masked to -inf")
#     for k in topk_list:
#         topk_idx = torch.topk(scores, k, dim=1).indices  # [B, k]
#         hit = (topk_idx == tgt_idx.view(-1,1)).any(dim=1).float()
#         # Recall@K for single-label = Hit@K
#         metrics[f'recall@{k}'] = hit.mean().item()
#         # Precision@K for single-label = Hit@K / K (macro-averaged)
#         metrics[f'precision@{k}'] = (hit / k).mean().item()
#         # MAP@K, NDCG@K
#         # find rank position if hit else inf
#         # positions in [0..k-1] where target occurs
#         pos = (topk_idx == tgt_idx.view(-1,1)).float().argmax(dim=1)
#         found = hit.bool()
#         # AP for single label = 1 / (rank+1) if found else 0
#         ap = torch.zeros(B)
#         ap[found] = 1.0 / (pos[found].float() + 1.0)
#         metrics[f'map@{k}'] = ap.mean().item()
#         # DCG for single label at rank r is 1/log2(r+2)
#         dcg = torch.zeros(B)
#         dcg[found] = 1.0 / torch.log2(pos[found].float() + 2.0)
#         # IDCG is 1.0
#         metrics[f'ndcg@{k}'] = dcg.mean().item()
#     return metrics