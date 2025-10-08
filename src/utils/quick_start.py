# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
from utils.recbole_eval import _pad_sequences, _strict_twostep_eval_sasfamily, _strict_twostep_eval_general, _strict_twostep_eval_context, rebuild_rb_inter
import platform
import traceback, os, shutil
import csv
import time
import torch
from recbole.quick_start import run_recbole
from recbole.config import Config as RBConfig
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils import get_model as rb_get_model
import glob, re, json
from tqdm.auto import tqdm
import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm

def _ensure_bucket_header(csv_path, bucket_labels):
    base = ['model', 'hyperparameters']
    extra = []
    for label in bucket_labels:
        for m in ('recall', 'ndcg'):
            for k in (10, 20):
                extra.append(f"test_{m}@{k}_{label}")
    header = base + extra
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(header)
    return header

def _eval_loader_for_bucket(config, train_ds, eval_ds, uid_field, iid_field, lo, hi, EvalDataLoader):
    # count training interactions per user
    hist_len = train_ds.df.groupby(uid_field)[iid_field].size()
    keep_users = hist_len[(hist_len >= lo) & (hist_len <= hi)].index.values
    sub_eval_df = eval_ds.df[eval_ds.df[uid_field].isin(keep_users)]
    if sub_eval_df.empty:
        return None, 0
    # use Dataset.copy to preserve meta (user_num, item_num, subseq maps)
    sub_eval_ds = eval_ds.copy(sub_eval_df)
    sub_eval_ds.inter_num = len(sub_eval_df)
    loader = EvalDataLoader(config, sub_eval_ds, additional_dataset=train_ds,
                            batch_size=config['eval_batch_size'])
    return loader, len(keep_users)

def quick_start(models, dataset, config_dict, save_model=True, mg=False):
    # Initialize CSV file
    # csv_path = 'multimodal_computation.csv'
    # if not os.path.exists(csv_path):
    #     with open(csv_path, 'a', newline='') as f:
    #         csv.writer(f).writerow(
    #             ['model_name', 'total_params', 'train_epoch_time','test_epoch_time', 'gpu_peak_mem_mb']
    #         )
    # csv_file = f'test_baby.csv'
    # if not os.path.exists(csv_file):
    #     with open(csv_file, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         # Write the header
    #         header = ['model', 'hyperparameters', 'valid_results', 'test_results']
    #         writer.writerow(header)

    overall_best_model_name = None
    overall_best_hyperparams = None
    overall_best_valid_results = None
    overall_best_test_results = None
    overall_best_value = float('-inf')
    val_metric = None  # Will be set after loading config

    for model_name in models:
        # Re-initialize config for each model
        config = Config(model_name, dataset, config_dict, mg)
        dataset_name = config['dataset']
        csv_file = f'multimodal_baselines_{dataset_name}.csv'
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write the header
                header = ['model', 'hyperparameters', 'valid_results', 'test_results']
                writer.writerow(header)

        bucket_csv = f"multimodal_{dataset_name}_userbucket_eval.csv"
        USER_HISTORY_BUCKETS = [
            ("1-5", 1, 5),
            ("6-20", 6, 20),
            ("21+", 21, 10 ** 9),
        ]
        bucket_header = _ensure_bucket_header(bucket_csv, [b[0] for b in USER_HISTORY_BUCKETS])
        # config['epochs'] = 1          # stop after a single epoch for timing

        init_logger(config)
        logger = getLogger()
        # print config info
        logger.info('██Server: \t' + platform.node())
        logger.info('██Dir: \t' + os.getcwd() + '\n')
        logger.info(config)

        # Set validation metric if not already set
        if val_metric is None:
            val_metric = config['valid_metric'].lower()

        RECB_SEQ_MODELS = {"SASRec", "BERT4Rec", "GRU4Rec", "FEARec", "SRGNN"}
        RECB_GENERAL_MODELS = {"SGL", "NeuMF", "MF"}
        RECB_CONTEXT_MODELS = {"FM", "FFM", "DeepFM", "xDeepFM", "DCN", "AutoInt", "FiBiNet"}
        if model_name in (RECB_SEQ_MODELS | RECB_GENERAL_MODELS | RECB_CONTEXT_MODELS):
            src_dir = os.path.dirname(__file__)
            repo_root = os.path.dirname(src_dir)  # .../src
            data_root = os.path.abspath(os.path.join(repo_root, "..", "data"))
            rb_dataset, typed_cols = rebuild_rb_inter(data_root, dataset, sort_for_repro=False)

            # infer field names for overrides:
            def pick_name(suffix):
                for ct in typed_cols:
                    n, *_ = ct.split(":", 1)
                    if suffix in n.lower():
                        return n
                return {"user": "userID", "item": "itemID", "time": "timestamp"}[suffix]

            USER_F = pick_name("user")
            ITEM_F = pick_name("item")
            TIME_F = pick_name("time")

            # 2) train with RecBole (standard LOO, full-ranking). Do NOT return after.
            model_yaml = os.path.join(repo_root, "configs", "model", f"{model_name.lower()}.yaml")
            data_yaml = os.path.join(repo_root, "configs", "dataset", f"{dataset}.yaml")
            config_file_list = [p for p in (model_yaml, data_yaml) if os.path.isfile(p)]
            run_tag = f"{model_name}-{rb_dataset}-{int(time.time())}"
            ckpt_dir = os.path.join("saved", run_tag)
            os.makedirs(ckpt_dir, exist_ok=True)

            rb_overrides = {
                "data_path": data_root,
                "show_progress": True,
                "console_logger_level": "INFO",
                "field_separator": "\t",
                "encoding": "utf-8",
                "USER_ID_FIELD": USER_F,
                "ITEM_ID_FIELD": ITEM_F,
                "TIME_FIELD": TIME_F,
                "load_col": {"inter": [USER_F, ITEM_F, TIME_F]},
                "ITEM_LIST_LENGTH_FIELD": "item_length",
                "LIST_SUFFIX": "_list",
                "MAX_ITEM_LIST_LENGTH": 50,
                # "loss_type": "CE",
                # "train_neg_sample_args": None,
                "loss_type": "BPR",
                "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},  # creates 'neg_itemID'
                "NEG_PREFIX": "neg_",
                "seed": 999,
                "reproducibility": True,
                "learner": "adam",
                "learning_rate": 1e-3,
                "epochs": 1,
                "stopping_step": 20,
                "train_batch_size": 512,
                "eval_batch_size": 4096,
                "eval_args": {  # one-step LOO *for training/selection only*
                    "group_by": "user",
                    "order": "TO",
                    "split": {"LS": "valid_and_test"},
                    "mode": {"valid": "full", "test": "full"}
                },
                "metrics": ["Recall", "NDCG", "Precision", "MAP"],
                "topk": [5, 10, 20, 50],
                "valid_metric": "Recall@20",
                "valid_metric_bigger": True,
                "checkpoint_dir": ckpt_dir
            }

            # if torch.cuda.is_available():
            #     torch.cuda.reset_peak_memory_stats(device=rb_overrides.get('device', 'cuda'))
            # t0 = time.perf_counter()

            _rb_res = run_recbole(
                model=model_name,
                dataset=rb_dataset,
                config_file_list=config_file_list,
                config_dict=rb_overrides,
                saved=True,
            )
            # train_epoch_time = time.perf_counter() - t0
            # if torch.cuda.is_available():
            #     peak_mem_bytes = torch.cuda.max_memory_allocated(device=rb_overrides.get('device', 'cuda'))
            #     gpu_peak_mem_mb = peak_mem_bytes / (1024 ** 2)
            # else:
            #     gpu_peak_mem_mb = 0.0

            best_valid_score = _rb_res["best_valid_score"]
            rb_valid_result = _rb_res["best_valid_result"]
            _rb_test_result = _rb_res["test_result"]

            tqdm.write(f"[tqdm] finished RecBole: {model_name}")

            # 3) locate latest checkpoint
            best_ckpt = _rb_res.get("saved_model_file")
            if not best_ckpt:
                ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
                assert ckpts, f"No checkpoints found in {ckpt_dir}"
                best_ckpt = ckpts[-1]

            # 4) rebuild RecBole context to load model
            rb_cfg = RBConfig(model=model_name, dataset=rb_dataset,
                              config_file_list=config_file_list, config_dict=rb_overrides)
            rb_dataset_obj = create_dataset(rb_cfg)  # reads {rb_dataset}.inter
            pad_id = 0
            tqdm.write(f"[strict] using pad_id={pad_id}")
            train_data, valid_data, test_data = data_preparation(rb_cfg, dataset=rb_dataset_obj)
            RBModel = rb_get_model(model_name)
            rb_model = RBModel(rb_cfg, train_data.dataset).to(rb_cfg["device"])
            state = torch.load(best_ckpt, map_location=rb_cfg["device"])
            rb_model.load_state_dict(state["state_dict"] if "state_dict" in state else state["model_state_dict"])
            rb_model.eval()
            # if torch.cuda.is_available():
            #     # Ensure all GPU work is finished before sampling memory
            #     torch.cuda.synchronize(rb_cfg['device'])
            #     peak_mem_bytes = torch.cuda.max_memory_allocated(device=rb_cfg['device'])
            #     gpu_peak_mem_mb = peak_mem_bytes / (1024 ** 2)
            # total_params = sum(p.numel() for p in rb_model.parameters() if p.requires_grad)
            # #
            # with open(csv_path, 'a', newline='') as f:
            #     csv.writer(f).writerow([
            #         model_name,
            #         f"{total_params:.2f}",
            #         0,0,
            #         f"{gpu_peak_mem_mb:.2f}",
            #     ])

            # 5) build strict two-step prefixes from the RecBole's INTERNAL Interaction
            #    (avoids any token/id mismatch)
            import pandas as pd
            uid_field = rb_dataset_obj.uid_field  # e.g., "userID"
            iid_field = rb_dataset_obj.iid_field  # e.g., "itemID"
            time_field = rb_dataset_obj.time_field  # e.g., "timestamp"

            def _to_df(split_data, extra_fields=None):
                inter = split_data.dataset.inter_feat
                cols = [uid_field, iid_field, time_field]
                if extra_fields:
                    # only include those that exist in this Interaction
                    existing = []
                    if hasattr(inter, "interaction"):
                        existing = [f for f in extra_fields if f in inter.interaction]
                        data = {f: inter.interaction[f].cpu().numpy() for f in (cols + existing)}
                    else:
                        existing = [f for f in extra_fields if f in inter]
                        data = {f: inter[f].cpu().numpy() for f in (cols + existing)}
                else:
                    if hasattr(inter, "interaction"):
                        data = {f: inter.interaction[f].cpu().numpy() for f in cols}
                    else:
                        data = {f: inter[f].cpu().numpy() for f in cols}
                import pandas as pd
                return pd.DataFrame(data).sort_values([uid_field, time_field])

            # 1) Try feat_name_list (can be dict or list depending on RecBole version)
            feat_name_list = getattr(rb_dataset_obj, "feat_name_list", None)

            inter_fields_all = None
            if isinstance(feat_name_list, dict):
                # modern: {'user': [...], 'item': [...], 'inter': [...]}
                inter_fields_all = list(feat_name_list.get("inter", []))
            elif isinstance(feat_name_list, list):
                # some builds expose a flat list of all fields; we’ll filter later
                inter_fields_all = list(feat_name_list)

            # 2) If still unknown, fall back to reading the Interaction keys directly
            if not inter_fields_all:
                _inter = train_data.dataset.inter_feat
                # Interaction can expose .interaction (dict) or be iterable over field names
                if hasattr(_inter, "interaction"):
                    inter_fields_all = list(_inter.interaction.keys())
                else:
                    try:
                        inter_fields_all = list(_inter.keys())
                    except Exception:
                        # last resort: probe attributes
                        inter_fields_all = [k for k in dir(_inter) if not k.startswith("_")]

            # 3) Build context candidates by excluding base & sequential fields
            base_exclude = {
                uid_field,
                iid_field,
                time_field,
                getattr(rb_model, "ITEM_SEQ", None),
                getattr(rb_model, "ITEM_SEQ_LEN", None),
            }
            inter_fields_all = [f for f in inter_fields_all if f]  # drop Nones/empties
            ctx_candidates = [
                f for f in inter_fields_all
                if f not in base_exclude and not str(f).startswith("neg_")
            ]

            train_df = _to_df(train_data, extra_fields=ctx_candidates).sort_values([uid_field, time_field])
            valid_df = _to_df(valid_data, extra_fields=ctx_candidates).sort_values([uid_field, time_field])
            test_df = _to_df(test_data, extra_fields=ctx_candidates).sort_values([uid_field, time_field])

            from collections import Counter
            cnt_train = Counter(train_df[uid_field].tolist())
            cnt_val = Counter(valid_df[uid_field].tolist())  # should be 1 per user if LS:valid_and_test
            cnt_test = Counter(test_df[uid_field].tolist())  # should be 1 per user

            users_all = sorted(set(cnt_train) | set(cnt_val) | set(cnt_test))
            tot_len = []
            for u in users_all:
                # In RecBole sequential, train_df usually has (n-2) rows per user; add 2 for val+test
                tot_len.append(cnt_train.get(u, 0) + cnt_val.get(u, 0) + cnt_test.get(u, 0))
            import numpy as np
            tot_len = np.array(tot_len)
            print("[stats] per-user total interactions (train+val+test) ->",
                  "min:", tot_len.min(), "p50:", np.percentile(tot_len, 50),
                  "p90:", np.percentile(tot_len, 90), "max:", tot_len.max())
            print("[stats] users with n<=2:", (tot_len <= 2).sum(),
                  "  n==2:", (tot_len == 2).sum(),
                  "  n>=3:", (tot_len >= 3).sum())

            device = rb_cfg["device"]
            item_num = rb_dataset_obj.item_num
            max_len = rb_overrides["MAX_ITEM_LIST_LENGTH"]

            # VALID target per user (penultimate)
            val_targets = {int(u): int(g[iid_field].iloc[-1]) for u, g in valid_df.groupby(uid_field)}
            # TEST target per user (last)
            test_targets = {int(u): int(g[iid_field].iloc[-1]) for u, g in test_df.groupby(uid_field)}

            # Build TRAIN history:
            # (A) sequential/session models → use RecBole-provided ITEM_SEQ/ITEM_SEQ_LEN at VAL
            use_seq = model_name in RECB_SEQ_MODELS and hasattr(rb_model, "ITEM_SEQ") and hasattr(rb_model,
                                                                                                  "ITEM_SEQ_LEN")
            # t1 = time.perf_counter()
            if use_seq:

                val_inter = valid_data.dataset.inter_feat
                seq_field = rb_model.ITEM_SEQ
                len_field = rb_model.ITEM_SEQ_LEN

                uid_arr = val_inter[uid_field].cpu().numpy()
                seq_mat = val_inter[seq_field]  # [B, L] padded with PAD=0
                len_vec = val_inter[len_field]  # [B]
                lens_rb = val_inter[len_field].cpu().numpy().astype(int)
                print("[RB] valid rows:", len(len_vec),
                      " usable (len>0):", (len_vec > 0).sum(),
                      " unusable (len==0):", (len_vec == 0).sum())

                train_hist = {}
                for i, u in enumerate(uid_arr):
                    L = lens_rb[i]
                    if L <= 0:
                        train_hist[int(u)] = []
                        continue
                    # Take exactly the last L tokens (they are non-zero by definition)
                    seq = seq_mat[i, :L].tolist()
                    train_hist[int(u)] = seq

                print("[rehydrate] from RB: B=", len(uid_arr),
                      " your_nonempty=", sum(1 for s in train_hist.values() if len(s) > 0),
                      " min_len=", min((len(s) for s in train_hist.values() if len(s) > 0), default=None))

                lens_you = np.array([len(train_hist[int(u)]) for u in uid_arr], dtype=int)
                mismatch = (lens_rb != lens_you)
                print("[cmp] RB lens>0:", (lens_rb > 0).sum(), "  your lens>0:", (lens_you > 0).sum(),
                      "  mismatches:", mismatch.sum())
                # show a few mismatches
                for i in np.where(mismatch)[0][:5]:
                    u = int(uid_arr[i])
                    print(
                        f"[cmp.ex] uid={u} RB_len={lens_rb[i]}  your_len={lens_you[i]}  your_tail={train_hist[u][-8:] if train_hist[u] else []}")
                assert mismatch.sum() == 0, "Rehydrated prefix lengths must match RecBole's ITEM_SEQ_LEN exactly"

                # ---- compute users_eval NOW (after train_hist exists, before using it) ----
                users_eval = sorted(set(train_hist.keys()) & set(val_targets.keys()) & set(test_targets.keys()))

                # ---- then derive the eval-sliced structures ----
                user2seq = {u: train_hist[u] for u in users_eval}  # prefixes for evaluator
                train_pos = {u: set(train_hist[u]) for u in users_eval}  # (optional) positives for masking

                print("[sanity] users_eval:", len(users_eval))
                prefix_lens = np.array([len(user2seq[u]) for u in users_eval], dtype=int)
                if len(prefix_lens) > 0:
                    print("[sanity] prefix_len stats -> min:", prefix_lens.min(),
                          "p50:", np.percentile(prefix_lens, 50),
                          "p90:", np.percentile(prefix_lens, 90),
                          "max:", prefix_lens.max())
                    b15 = ((prefix_lens >= 1) & (prefix_lens <= 5)).sum()
                    b620 = ((prefix_lens >= 6) & (prefix_lens <= 20)).sum()
                    b21p = (prefix_lens >= 21).sum()
                    print(f"[sanity] bucket user counts -> 1-5:{b15}  6-20:{b620}  21+:{b21p}")
                else:
                    print("[sanity] prefix_len stats -> EMPTY")

                same_ct = sum(1 for u in users_eval if val_targets[u] == test_targets[u])
                print(f"[sanity] val==test target for {same_ct}/{len(users_eval)} users")
                # show 3 long examples if available
                for u in [u for u in users_eval if len(user2seq[u]) >= 6][:3]:
                    print(
                        f"[ex] uid={u} len={len(user2seq[u])} tail={user2seq[u][-8:]} val={val_targets[u]} test={test_targets[u]}")

                for u in [u for u in users_eval if 3 <= len(user2seq[u]) <= 12][:3]:
                    print(f"[ex.fixed] uid={u} len={len(user2seq[u])} tail8={user2seq[u][-8:]}")

                nz_users = [u for u in users_eval if len(user2seq[u]) > 0]
                print("[evalset] users_eval:", len(users_eval), "  nonempty_prefix:", len(nz_users))
                # assert we evaluate all non-empty prefixes
                assert len(nz_users) == len(user2seq), "user2seq unexpectedly filtered elsewhere"

            else:
                train_pos = {int(u): set(g[iid_field].tolist()) for u, g in train_df.groupby(uid_field)}
                users_eval = sorted(set(train_pos.keys()) & set(val_targets.keys()) & set(test_targets.keys()))
                user2seq = None  # not used

            # ---------- Context tensors (only if model is context-aware) ----------
            def _build_ctx_tensors(df, users):
                # build last-row context per user for all candidate context fields
                if not ctx_candidates:
                    return {}
                # decide dtype from Dataset.field2type if available
                f2type = getattr(rb_dataset_obj, "field2type", {})
                tensors = {}
                for f in ctx_candidates:
                    if f not in df.columns:
                        continue
                    series = df.groupby(uid_field)[f].last()  # per-user last value (VAL or TEST)
                    vals = [series.get(u, 0) for u in users]
                    # token-like → long; float-like → float
                    is_token = str(f2type.get(f, "")).lower().find("token") >= 0
                    import torch
                    tensors[f] = torch.tensor(vals, dtype=torch.long if is_token else torch.float32, device=device)
                return tensors

            ctx_valid = _build_ctx_tensors(valid_df, users_eval) if model_name in RECB_CONTEXT_MODELS else {}
            ctx_test = _build_ctx_tensors(test_df, users_eval) if model_name in RECB_CONTEXT_MODELS else {}

            # ---------- Strict two-step evaluation by model family ----------
            if use_seq:
                eval_chunk = 4096  # safe micro-batch for SAS/BERT/FEA/SRGNN
                strict_valid = _strict_twostep_eval_sasfamily(
                    rb_model, device, item_num, {u: user2seq[u] for u in users_eval if u in user2seq},
                    train_pos, val_targets, max_len=max_len, pad_id=pad_id, batch_users=eval_chunk
                )
                strict_test = _strict_twostep_eval_sasfamily(
                    rb_model, device, item_num, {u: user2seq[u] for u in users_eval if u in user2seq},
                    train_pos, test_targets, max_len=max_len, pad_id=pad_id, stage='test', val_targets=val_targets,
                    batch_users=eval_chunk
                )
            elif model_name in RECB_GENERAL_MODELS:
                eval_chunk_general = 4096  # safe micro-batch size for SGL/LightGCN/BPR
                strict_valid = _strict_twostep_eval_general(
                    rb_model, device, item_num, users_eval, train_pos, val_targets,
                    stage="valid", batch_users=eval_chunk_general
                )
                strict_test = _strict_twostep_eval_general(
                    rb_model, device, item_num, users_eval, train_pos, test_targets,
                    stage="test", val_targets=val_targets, batch_users=eval_chunk_general
                )
            else:  # context-aware
                strict_valid = _strict_twostep_eval_context(
                    rb_model, device, item_num, users_eval, train_pos, val_targets, ctx_valid, stage="valid"
                )
                strict_test = _strict_twostep_eval_context(
                    rb_model, device, item_num, users_eval, train_pos, test_targets, ctx_test, stage="test",
                    val_targets=val_targets
                )

            # 6) write row to recbole_baselines_{dataset}.csv
            base_csv = f"recbole_baselines_{dataset}.csv"
            if not os.path.exists(base_csv):
                with open(base_csv, "w", newline="") as f:
                    csv.writer(f).writerow(["model", "hyperparameters", "valid_results", "test_results"])
            hyperparams_str = {"seed": rb_overrides["seed"], "loss_type": rb_overrides["loss_type"]}
            with open(base_csv, "a", newline="") as f:
                csv.writer(f).writerow([model_name,
                                        json.dumps(hyperparams_str, separators=(',', ':')),
                                        json.dumps(strict_valid, separators=(',', ':')),
                                        json.dumps(strict_test, separators=(',', ':'))])

            # 7) user-bucket evaluation (like your other models)
            bucket_csv = f"baselines_{dataset}_userbucket_eval.csv"
            USER_HISTORY_BUCKETS = [("1-5", 1, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 9)]
            if not os.path.exists(bucket_csv):
                hdr = ["model", "hyperparameters"]
                for label in [b[0] for b in USER_HISTORY_BUCKETS]:
                    for m in ("recall", "ndcg"):
                        for k in (10, 20):
                            hdr.append(f"test_{m}@{k}_{label}")
                with open(bucket_csv, "w", newline="") as f:
                    csv.writer(f).writerow(hdr)

            # compute training history length (strict: up to t_{n-2})
            # len up to t_{n-2} equals max(0, len(seq)-2), bucket by that proxy
            if use_seq:
                uid2len = {int(u): len(user2seq.get(int(u), ())) for u in users_eval}
            else:
                uid2len = {int(u): len(train_pos.get(int(u), ())) for u in users_eval}
            bucket_values = {
                b[0]: {"recall@10": "", "recall@20": "", "ndcg@10": "", "ndcg@20": ""}
                for b in USER_HISTORY_BUCKETS
            }

            for (label, lo, hi) in USER_HISTORY_BUCKETS:
                u_keep = [u for u, L in uid2len.items() if lo <= L <= hi]
                if not u_keep:
                    continue
                if use_seq:
                    sub_user2seq = {u: user2seq[u] for u in u_keep if u in user2seq}
                    sub_train_pos = {u: train_pos[u] for u in u_keep}
                    sub_targets = {u: test_targets[u] for u in u_keep}
                    res = _strict_twostep_eval_sasfamily(
                        rb_model, device, item_num,
                        sub_user2seq, sub_train_pos, sub_targets,
                        max_len=max_len, pad_id=pad_id,
                        topk_list=(10, 20), stage="test", val_targets=val_targets
                    )
                elif model_name in RECB_GENERAL_MODELS:
                    sub_train_pos = {u: train_pos[u] for u in u_keep}
                    sub_targets = {u: test_targets[u] for u in u_keep}
                    res = _strict_twostep_eval_general(
                        rb_model, device, item_num, u_keep, sub_train_pos, sub_targets, stage="test",
                        val_targets=val_targets, topk_list=(10, 20)
                    )
                else:
                    sub_train_pos = {u: train_pos[u] for u in u_keep}
                    sub_targets = {u: test_targets[u] for u in u_keep}

                    idx_map = {u: i for i, u in enumerate(users_eval)}

                    def _subset_ctx(ctx):
                        if not ctx:
                            return {}
                        import torch
                        idxs = torch.tensor([idx_map[u] for u in u_keep], dtype=torch.long,
                                            device=next(iter(ctx.values())).device)
                        return {f: t.index_select(0, idxs) for f, t in ctx.items()}

                    res = _strict_twostep_eval_context(
                        rb_model, device, item_num, u_keep, sub_train_pos, sub_targets,
                        _subset_ctx(ctx_test), stage="test", val_targets=val_targets, topk_list=(10, 20)
                    )

                if res:
                    bucket_values[label]["recall@10"] = float(res.get("recall@10", ""))
                    bucket_values[label]["recall@20"] = float(res.get("recall@20", ""))
                    bucket_values[label]["ndcg@10"] = float(res.get("ndcg@10", ""))
                    bucket_values[label]["ndcg@20"] = float(res.get("ndcg@20", ""))

            labels_in_order = [b[0] for b in USER_HISTORY_BUCKETS]
            row = [model_name, json.dumps(hyperparams_str, separators=(",", ":"))]
            for lbl in labels_in_order:
                for m in ("recall", "ndcg"):
                    for k in (10, 20):
                        row.append(bucket_values[lbl][f"{m}@{k}"])

            # sanity: row length should be 2 + 3 buckets * 4 metrics = 14
            expected_cols = 2 + len(labels_in_order) * 4
            assert len(row) == expected_cols, f"[bucket write] wide row has {len(row)} cols, expected {expected_cols}"

            with open(bucket_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print("[bucket write row] ok. counts:",
                  {lbl: sum(1 for _ in [bucket_values[lbl]["recall@10"]] if bucket_values[lbl]["recall@10"] != "")
                   for lbl in labels_in_order})

            continue
        # Load data
        dataset_obj = RecDataset(config)
        # print dataset statistics
        logger.info(str(dataset_obj))

        train_dataset, valid_dataset, test_dataset = dataset_obj.split()
        logger.info('\n====Training====\n' + str(train_dataset))
        logger.info('\n====Validation====\n' + str(valid_dataset))
        logger.info('\n====Testing====\n' + str(test_dataset))

        # Wrap into dataloader
        train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
        (valid_data, test_data) = (
            EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
            EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

        logger.info(f'\n\n========== Starting evaluation for model: {model_name} ==========\n')

        hyper_ret = []
        best_valid_value = float('-inf')
        best_test_value = float('-inf')
        best_hyperparams = None
        best_valid_results = None
        best_test_results = None
        best_model = None  # To store the best model instance

        # Hyper-parameters
        hyper_ls = []
        if "seed" not in config['hyper_parameters']:
            config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
        for param in config['hyper_parameters']:
            param_values = config[param]
            if not isinstance(param_values, list):
                param_values = [param_values]
            hyper_ls.append(param_values)
        # Combinations
        combinators = list(product(*hyper_ls))
        total_loops = len(combinators)

        idx = 0

        for hyper_tuple in combinators:
            # Update hyperparameters in config
            for param_name, param_value in zip(config['hyper_parameters'], hyper_tuple):
                config[param_name] = param_value
            init_seed(config['seed'])

            logger.info('========={}/{}: Parameters:{}={}======='.format(
                idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

            # Set random state of dataloader
            train_data.pretrain_setup()
            # Model loading and initialization
            if 'ssg' in config['model'].lower() or 'sgcl' in config['model'].lower():
                model = get_model(config['model'])(config, train_data, train_dataset).to(config['device'])
            else:
                model = get_model(config['model'])(config, train_data).to(config['device'])
            logger.info(model)

            # Trainer loading and initialization
            trainer = get_trainer()(config, model, mg)

            # ── computation probe (multimodal only) ──────────────────────
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            #
            # # Reset peak mem so the final read includes train + val + test
            # if torch.cuda.is_available():
            #     torch.cuda.reset_peak_memory_stats(device=config['device'])
            #
            # # --- temporarily disable validation/test INSIDE fit so train time is pure ---
            # orig_valid_epoch = getattr(trainer, '_valid_epoch', None)
            # orig_evaluate = getattr(trainer, 'evaluate', None)
            #
            # def _noop_valid_epoch(_data):
            #     # return the same structure your fit expects: (score, result-dict)
            #     return 0.0, {}
            #
            # def _noop_evaluate(_data, *args, **kwargs):
            #     # return empty result dict
            #     return {}
            #
            # try:
            #     if orig_valid_epoch is not None:
            #         trainer._valid_epoch = _noop_valid_epoch
            #     if orig_evaluate is not None:
            #         trainer.evaluate = _noop_evaluate
            #
            #     # === Pure TRAIN epoch time (no val/test work inside) ===
            #     t0 = time.perf_counter()
            #     _ = trainer.fit(
            #         train_data, valid_data=valid_data, test_data=None, saved=save_model
            #     )
            #     if torch.cuda.is_available():
            #         torch.cuda.synchronize()  # ensure GPU kernels finished
            #     train_epoch_time = time.perf_counter() - t0
            # finally:
            #     # restore real validation/test methods
            #     if orig_valid_epoch is not None:
            #         trainer._valid_epoch = orig_valid_epoch
            #     if orig_evaluate is not None:
            #         trainer.evaluate = orig_evaluate
            #
            # # (Optional) run a proper validation pass to have metrics for your logs/CSV (not timed)
            # best_valid_result = trainer.evaluate(valid_data, is_test=False)
            #
            # # === Pure TEST epoch time ===
            # t1 = time.perf_counter()
            # best_test_upon_valid = trainer.evaluate(test_data, is_test=True)
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            # test_epoch_time = time.perf_counter() - t1
            #
            # # Peak memory since the reset → includes train + val + test
            # if torch.cuda.is_available():
            #     peak_mem_bytes = torch.cuda.max_memory_allocated(device=config['device'])
            #     gpu_peak_mem_mb = peak_mem_bytes / (1024 ** 2)
            # else:
            #     gpu_peak_mem_mb = 0.0
            #
            # # Write rounded to 2 d.p. (multimodal only CSV)
            # with open(csv_path, 'a', newline='') as f:
            #     csv.writer(f).writerow([
            #         model_name,
            #         f"{total_params:.2f}",
            #         f"{train_epoch_time:.2f}",
            #         f"{test_epoch_time:.2f}",
            #         f"{gpu_peak_mem_mb:.2f}",
            #     ])
            # # ─────────────────────────────────────────────────────────────
            # Model training

            vals = []
            for v in hyper_tuple:
                s = str(v).replace('/', '-').replace(' ', '')  # light sanitization
                vals.append(s)
            config['checkpoint_name_tag'] = ".".join(vals)[:80]

            # best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            #     train_data, valid_data=valid_data, test_data=test_data, saved=save_model)

            if save_model and getattr(trainer, "best_ckpt_path", None):
                state = torch.load(trainer.best_ckpt_path, map_location=config['device'])
                model.load_state_dict(state['model_state_dict'])

            # NEW — run bucketed TEST evaluation
            uid_field, iid_field = train_dataset.uid_field, train_dataset.iid_field
            bucket_metrics = {}
            for (label, lo, hi) in USER_HISTORY_BUCKETS:
                loader, nunique = _eval_loader_for_bucket(
                    config, train_dataset, test_dataset, uid_field, iid_field, lo, hi, EvalDataLoader
                )
                if loader is None:
                    for m in ('recall', 'ndcg'):
                        for k in (10, 20):
                            bucket_metrics[f"test_{m}@{k}_{label}"] = ""
                    continue
                res = trainer.evaluate(loader, is_test=True)  # standard full-sort evaluator
                for m in ('recall', 'ndcg'):
                    for k in (10, 20):
                        bucket_metrics[f"test_{m}@{k}_{label}"] = res.get(f"{m}@{k}", "")

            # NEW — append a row to the user-bucket CSV
            with open(bucket_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                hyperparams_str = dict(zip(config['hyper_parameters'], hyper_tuple))
                row = [model_name, hyperparams_str]
                for col in bucket_header[2:]:
                    row.append(bucket_metrics.get(col, ""))
                writer.writerow(row)

            hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

            # Save best validation result
            current_valid_value = best_valid_result[val_metric]
            if current_valid_value > best_valid_value:
                best_valid_value = current_valid_value
                best_hyperparams = hyper_tuple  # Store the best hyperparameters
                best_valid_results = best_valid_result
                best_test_results = best_test_upon_valid
                if model_name.lower() == 'freedom':
                    best_model = model  # Store the best model instance for freedom

            # Save results to CSV after each hyperparameter configuration
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                hyperparams_str = dict(zip(config['hyper_parameters'], hyper_tuple))
                row = [model_name, hyperparams_str, dict2str(best_valid_result), dict2str(best_test_upon_valid)]
                writer.writerow(row)

            logger.info('Valid result: {}'.format(dict2str(best_valid_result)))
            logger.info('Test result: {}'.format(dict2str(best_test_upon_valid)))
            logger.info('████Current BEST for {}████:\nParameters: {},\n'
                        'Valid: {},\nTest: {}\n\n\n'.format(model_name, hyperparams_str,
                                                            dict2str(best_valid_results),
                                                            dict2str(best_test_results)))

            idx += 1

        # After evaluating all hyperparameters for the current model
        logger.info('\n============ Results for model: {} ==============='.format(model_name))
        for (p, k, v) in hyper_ret:
            hyperparams_str = dict(zip(config['hyper_parameters'], p))
            logger.info('Parameters: {},\n best valid: {},\n best test: {}'.format(hyperparams_str,
                                                                                   dict2str(k), dict2str(v)))

        logger.info('\n\n█████████████ BEST for {} ████████████████'.format(model_name))
        hyperparams_str = dict(zip(config['hyper_parameters'], best_hyperparams))
        logger.info('\tParameters: {},\nValid: {},\nTest: {}\n\n'.format(hyperparams_str,
                                                                         dict2str(best_valid_results),
                                                                         dict2str(best_test_results)))

        # Save embeddings if the model is 'freedom' and best_model exists
        if model_name.lower() == 'freedom' and best_model is not None:
            # Call the forward function with the adjacency matrix
            adj = best_model.norm_adj  # Get the adjacency matrix
            adj = adj.to(config['device'])  # Ensure it's on the correct device
            # Get the user and item embeddings
            user_embeddings, item_embeddings = best_model(adj)
            # Ensure that embeddings are detached and moved to CPU
            user_embeddings = user_embeddings.detach().cpu()
            item_embeddings = item_embeddings.detach().cpu()
            torch.save(user_embeddings, 'best_user_embeddings_clothing.pt')
            torch.save(item_embeddings, 'best_item_embeddings_clothing.pt')
            logger.info('User and item embeddings have been saved as .pt files for the FREEDOM model.')

        # Update overall best model if current model's best validation metric is better
        if best_valid_value > overall_best_value:
            overall_best_value = best_valid_value
            overall_best_hyperparams = best_hyperparams
            overall_best_valid_results = best_valid_results
            overall_best_test_results = best_test_results
            overall_best_model_name = model_name

    # Log the overall best model among all evaluated models
    if overall_best_model_name is not None:
        logger.info('\n============ Overall Best Model ===============')
        hyperparams_str = dict(zip(config['hyper_parameters'], overall_best_hyperparams))
        logger.info('Best Model: {}'.format(overall_best_model_name))
        logger.info('Parameters: {},\nValid: {},\nTest: {}\n'.format(
            hyperparams_str, dict2str(overall_best_valid_results), dict2str(overall_best_test_results)))
    else:
        logger.info('\n============ Overall Best Model ===============\n(no non-RecBole models evaluated)')

