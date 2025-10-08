# MuSTRec – Quick-start

Minimal instructions to get **MuSTRec** running on the four Amazon subsets **baby · clothing · sports · elec**.

---

## 1. Setup
```bash
# after downloading the repo
cd MuSTRec

# create env & install deps
python -m venv venv        # or conda
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Grab the data
1. Open **`data/README.md`** and follow the Google‑Drive link.  
2. Download the data for baby, clothing, sports and elec so you have all image\_feat.npy and text\_feat.npy for all the datasets and also:
   ```
   data/{baby,clothing,sports,elec}/{dataset}.inter
   ```

---

## 3. Pre‑process
```bash
# 3‑A  split – last 2 actions → val/test
# you need to go into the script and modify the file path for each dataset
python data/diff_split.py

# 3‑B  turn interactions into padded sequences (max_len=50)
# you need to go into the script and modify the file path to access the .inter files for each dataset.
python preprocessing/new_seq.py     --dataset baby     --input  data/raw/baby/baby_diff.inter
```
Create `{dataset}_new.txt` and `{dataset}_diff_split.inter` files for all datasets.

---

## 4. Tell MuSTRec where the `{dataset}_new.txt` files live
This needs to be edited manually in `src/utils/seq_dataset.py`
It should be set for baby dataset by default.


## 5. Train / evaluate
```bash
python src/main.py --dataset baby        # <- choose clothing / sports / elec
# optional extras
#   --model MuSTRec   # default
#   --mg              # default false
```

---
