# MuSICRec – Quick-start

Minimal instructions to get **MuSICRec** and all baselines running on the Amazon subsets **baby · sports · elec**.

---

## 1. Setup
```bash
# after downloading the repo
cd MuSICRec

# create env & install deps
python -m venv venv        # or conda
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Grab the data
1. Open **`data/README.md`** and follow the Google‑Drive link.  
2. Download the data for baby, sports and elec so you have all image\_feat.npy and text\_feat.npy for all the datasets. Interaction files included and yaml files configured to use them:
   ```
   data/{baby,sports,elec}/{dataset_diff_split}.inter
   ```

---

## 3. Pre‑process
```bash
# 3‑A  split – last 2 actions → val/test
# you need to go into the script and modify the file path for each dataset
python data/diff_split.py

# 3‑B  turn interactions into sequences
# you need to go into the script and modify the file path to access the .inter files for each dataset. This creates the sequences for each user to build sequence nodes.
python MuSICRec/data/baby/sequence_generate_new.py     
```
`{dataset}_diff_split.inter` and `{dataset}_sequence_train` files have been supplied.

---

## 4. Our model is named as SGCL in the repo. sgcl.py under MuSICRec/src/models contains MuSICRec implementation. SGCL.yaml contains hyperparameters and configurations.
This will be modified for publication


## 5. Train / evaluate
```bash
python src/main.py --dataset baby        # <- choose clothing / sports / elec
# optional extras
#   --model SGCL   # default
#   --config       # config override for ablation
```

---
