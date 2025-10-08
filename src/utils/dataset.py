# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']
        self.time_field = self.config['TIME_FIELD']

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

        if self.time_field:
            last_ts = self.df.groupby(self.uid_field)[self.time_field].max()
            self.user_last_ts = np.zeros(self.user_num, dtype=np.float32)
            self.user_last_ts[last_ts.index.values] = last_ts.values
        else:
            self.user_last_ts = np.zeros(self.user_num, dtype=np.float32)

        if self.time_field:
            ui_last = self.df.groupby([self.uid_field, self.iid_field])[self.time_field].max()
            # store as {(u,i): ts} dictionary for quick lookup
            self.ui_last_ts = {(u, i): t for (u, i), t in ui_last.items()}
        else:
            self.ui_last_ts = {}

        if 'ssg' in self.config['model'].lower() or 'sgcl' in self.config['model'].lower():
            subseq_file_path = os.path.join(
                self.dataset_path,
                f"{self.dataset_name}_sequences_train"
            )
            subseq_file_path = subseq_file_path#'/home/bs00826/Downloads/multimodal_testing_whyper/data/baby/baby_sequences_train'
            self.load_subseq_map(subseq_file_path)

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        if self.time_field:
            cols.append(self.time_field)
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        if hasattr(self, 'user_last_ts'):
            nxt.user_last_ts = self.user_last_ts
        if hasattr(self, 'ui_last_ts'):
            nxt.ui_last_ts = self.ui_last_ts
        #might need changing
        if hasattr(self, 'subseq_map'):
            nxt.subseq_map = self.subseq_map
        if hasattr(self, 'n_subseq'):
            nxt.n_subseq = self.n_subseq
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)

    def load_subseq_map(self, subseq_path):
        """
        Reads each line of baby_subsequences.txt, building a dictionary:
          { subseq_id: [item1, item2, ...] }
        """
        # subseq_path = os.path.join(self.dataset_path, subseq_file_name)
        # if not os.path.isfile(subseq_path):
        #     raise FileNotFoundError(f"Subsequence file {subseq_path} not found.")

        self.subseq_map = {}
        with open(subseq_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                s_id = int(tokens[0])
                items = list(map(int, tokens[1:]))
                self.subseq_map[s_id] = items

        # store how many subsequences were loaded
        self.n_subseq = len(self.subseq_map)
        self.logger.info(f"Loaded {self.n_subseq} subsequences from {subseq_path}.")

    def get_user_seq(self, uid, pad_val=0):
        """
        Return the list of item‑ids (as a NumPy array) that constitute
        the only sequence owned by user `uid`.
        """
        seq = self.subseq_map[uid]
        return np.array(seq, dtype=np.int64)
