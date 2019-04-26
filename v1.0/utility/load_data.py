#!/usr/bin/env python
import scipy.sparse as sp
import numpy as np


class Data(object):
    def __init__(self, path):
        self.path = path

        train_file = path + '/train.data'
        valid_file = path + '/valid.data'
        test_file = path + '/test.data'

        # get number of users and items & then load rating data from train_file & test_file.
        self.n_users, self.n_items, self.n_feats = 0, 0, 0
        self.n_train, self.n_valid, self.n_test = 0, 0, 0

        self.train_label, self.valid_label, self.exist_users = [], [], []
        self.train_items, self.valid_set, self.test_set = {}, {}, {}
        self.train_feats, self.valid_feats, self.test_feats = [], [], []

        self._statistic_ratings(train_file, valid_file, test_file)

        self.get_identity_mat()
        self.get_interaction_mat(train_file, valid_file, test_file)

        self.get_n_features()

    def _statistic_ratings(self, train_file, valid_file, test_file):
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(',')
                    items = int(l[2]) - 1
                    feats = [int(i) - 13 for i in l[3:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max([items]))
                    self.n_users = max(self.n_users, uid)
                    self.n_feats = max(self.n_feats, max(feats))
                    self.n_train += len([items])

        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(',')
                    items = int(l[2]) - 1
                    self.n_items = max(self.n_items, max([items]))
                    self.n_valid += len([items])

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(',')
                    items = int(l[2]) - 1
                    self.n_items = max(self.n_items, max([items]))
                    self.n_test += len([items])
        self.n_items += 1
        self.n_users += 1
        self.n_feats += 1

    def get_identity_mat(self):
        self.user_one_hot = sp.identity(self.n_users).tocsr()
        self.item_one_hot = sp.identity(self.n_items).tocsr()

    def get_interaction_mat(self, train_file, valid_file, test_file):
        # Convert the training & test user-item interactions to sparse matrices.
        def get_sp_mat(file_name):
            rows, cols, vals = [], [], []
            frows, fcols, fvals = [], [], []
            user_item_dict, user_label_dict = {}, {}

            with open(file_name) as f:
                for l in f.readlines():
                    if len(l) == 0: break

                    l = l.strip('\n')
                    items = [int(i) for i in l.split(',')]
                    uid, label, item, feat = items[0], items[1], items[2], items[3:]
                    item = item - 1
                    feat = list(map(lambda x: x - 13, feat))

                    for iid in [item]:
                        rows.append(uid)
                        cols.append(iid)
                        vals.append(1)
                    for fid in feat:
                        frows.append(uid)
                        fcols.append(fid)
                        fvals.append(1)
                    user_item_dict[uid] = [items[2]]
                    # user_item_dict[uid] = items
                    user_label_dict[uid] = label
                # user_label_dict = tolist(user_label_dict)
                sp_mat = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))
                sp_feat_mat = sp.coo_matrix((fvals, (frows, fcols)), shape=(self.n_users, self.n_feats)).tocsr()
            return sp_mat, list(user_label_dict.items()), user_item_dict, sp_feat_mat

        # Sparse training matrices;
        self.sp_train, self.train_label, self.train_items, self.train_feats = get_sp_mat(train_file)
        print('already load training interaction mat', self.sp_train.shape)
        # Sparse validation matrices;
        self.sp_valid, self.valid_label, self.valid_set, self.valid_feats = get_sp_mat(valid_file)
        print('already load validation interaction mat', self.sp_valid.shape)
        # Sparse test matrices;
        self.sp_test, _, self.test_set, self.test_feats = get_sp_mat(test_file)
        print('already load test interaction mat', self.sp_test.shape)

    def generate_sp_train_batch(self, batch_size):
        instance_idx = []
        while True:
            if len(instance_idx) == batch_size: break

            idx = np.random.randint(low=0, high=len(self.sp_train.row), size=1)[0]
            instance_idx.append(idx)
        # get the indices of users, positive items, and negative items.
        user_idx = self.sp_train.row[instance_idx]
        label_idx = self.sp_train.row[instance_idx]
        feat_row_idx = self.sp_train.row[instance_idx]
        item_idx = self.sp_train.col[instance_idx]
        # Convert from indices to one-hot matrices
        users = self.user_one_hot[user_idx]
        items = self.item_one_hot[item_idx]
        # get feature batch
        feat = []
        for row in feat_row_idx:
            feat_mat = self.train_feats.getrow(row)
            if feat == []:
                feat = feat_mat
            else:
                feat = sp.vstack([feat, feat_mat])
        feats = sp.hstack([items, feat])
        # get label for each instance
        label_array = np.array(self.train_label)
        labels = label_array[label_idx]
        labels = labels[:, 1]

        return (users, items, feats, labels)

    def get_n_features(self):
        print('[n_train, n_valid, n_test]=[%d, %d, %d]' %
              (len(self.sp_train.row), len(self.sp_valid.row), len(self.sp_test.row)))
        self.n_features = self.n_items + self.n_feats