#!/usr/bin/env python3
"""
Generate baby_sequences_train and baby_sequences_test from a .inter file
that lists one interaction per line:

    userID  itemID  rating  timestamp

Assumptions
-----------
* The file is already sorted by userID, then by timestamp (chronological).
* For every user the final two interactions are 'validation' and 'test'.

Output
------
baby_sequences_train : userID followed by ALL items except the last two
baby_sequences_test  : userID followed by ALL items except the last one
"""
import argparse
from pathlib import Path

def flush_user(uid, items, f_train, f_test):
    """Write aggregated sequences for a single user."""
    if not items:               # no interactions → nothing to write
        return

    # test sequence = everything except the last (test) interaction
    test_seq = items[:-1]
    if test_seq:                # need ≥1 item
        f_test.write(f"{uid} {' '.join(test_seq)}\n")

    # train sequence = everything except validation + test
    train_seq = items[:-2]
    if train_seq:               # might be empty if user has only 2 interactions
        f_train.write(f"{uid} {' '.join(train_seq)}\n")

def main(path_in: Path,
         path_train: Path,
         path_test: Path):

    with path_in.open() as fin, \
         path_train.open("w") as ftr, \
         path_test.open("w")  as fte:

        current_uid = None
        bucket      = []        # holds itemIDs for current user

        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue                      # skip blank / whitespace lines
            if raw.lower().startswith("userid"):
                continue                      # skip header row if present

            parts = raw.split()
            uid, item = parts[0], parts[1]    # ignore rating & timestamp

            # new user? — flush previous one
            if current_uid is not None and uid != current_uid:
                flush_user(current_uid, bucket, ftr, fte)
                bucket = []

            # accumulate itemID
            current_uid = uid
            bucket.append(item)

        # flush last user
        flush_user(current_uid, bucket, ftr, fte)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Create baby_sequences_{train,test} from baby_diff_split.inter")
    p.add_argument("-i", "--input",  default="../elec/elec_diff_split.inter",
                   help="input .inter file (default: %(default)s)")
    p.add_argument("-o", "--outstem", default="../elec/elec_sequences",
                   help="stem for output files (default: %(default)s)")
    args = p.parse_args()

    main(Path(args.input),
         Path(f"{args.outstem}_train"),
         Path(f"{args.outstem}_test"))

    print(f"✔  Wrote {args.outstem}_train and {args.outstem}_test")
