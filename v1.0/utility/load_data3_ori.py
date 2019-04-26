#!/usr/bin/env python
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import copy
import os
from time import time

class Data(object):
    def __init__(self, path):
        self.path = path

        train_file = path + '/train.dat'
        test_file = path + '/test.dat'
        test_neg_file = path + '/test_neg.dat'

        self.user_social_file = path + '/user_social.npz'
        self.user_side_file = path + '/user_side.npz'
        self.item_side_file = path + '/item_side.npz'

        # get number of users and items & then load rating data from train_file & test_file.
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0

        self.exist_users = []
        self.train_items, self.test_set = {}, {}

        self._statistic_ratings(train_file, test_file)

        self.get_identity_mat()
        self.get_interaction_mat(train_file, test_file, test_neg_file)
        self.get_user_social_mat()
        self.get_user_side_mat()
        self.get_item_side_mat()

        self.get_n_features()

    def _statistic_ratings(self, train_file, test_file):
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

    def get_identity_mat(self):
        self.user_one_hot = sp.identity(self.n_users).tocsr()
        self.item_one_hot = sp.identity(self.n_items).tocsr()

    def get_interaction_mat(self, train_file, test_file, test_neg_file):
        # Convert the training & test user-item interactions to sparse matrices.
        def get_sp_mat(file_name):
            rows, cols, vals = [], [], []
            user_item_dict = {}

            with open(file_name) as f:
                for l in f.readlines():
                    if len(l) == 0: break

                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, pos_items = items[0], items[1:]

                    for iid in pos_items:
                        rows.append(uid)
                        cols.append(iid)
                        vals.append(1)
                    user_item_dict[uid] = pos_items

                sp_mat = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))
            return sp_mat, user_item_dict

        # Sparse training matrices;
        self.sp_train, self.train_items = get_sp_mat(train_file)
        print('already load training interaction mat', self.sp_train.shape)
        # Sparse test matrices;
        self.sp_test, self.test_set = get_sp_mat(test_file)
        print('already load test interaction mat', self.sp_test.shape)
        # Sparse test negative matrices
        try:
            self.sp_test_neg, self.test_set_neg = get_sp_mat(test_neg_file)
        except Exception:
            self.create_neg_test_mat(test_neg_file, neg_num=10)
            self.sp_test_neg, self.test_set_neg = get_sp_mat(test_neg_file)
        print('already load test negative interaction mat', self.sp_test_neg.shape)

    def create_neg_test_mat(self, test_neg_file, neg_num=10):
        f = open(test_neg_file, 'w')

        for u_id in self.test_set.keys():
            training_items = self.train_items[u_id]
            test_pos_items = self.test_set[u_id]
            test_neg_items = []


            while True:
                if len(test_neg_items) == neg_num * len(test_pos_items): break
                neg_id = np.random.randint(low=0, high=self.n_items - 1, size=1)[0]
                if neg_id not in training_items and neg_id not in test_pos_items and neg_id not in test_neg_items:
                    test_neg_items.append(neg_id)

            f.write(str(u_id) + ' ' + ' '.join([str(i) for i in test_neg_items]) + '\n')
        f.close()
        print('already generate test negative interaction data')

    def get_user_social_mat(self):
        try:
            user_social_mat = sp.load_npz(self.user_social_file)
            print('already load user social mat', user_social_mat.shape)
        except Exception:
            user_social_mat = None
            print('social mat is none.')
        # self.user_social_mat = None
        self.user_social_mat = user_social_mat

    def get_user_side_mat(self):
        try:
            user_side_mat = sp.load_npz(self.user_side_file)
            self.n_u_cats = user_side_mat.shape[1]
            print('already load user side mat', user_side_mat.shape)
        except Exception:
            user_side_mat = None
            self.n_u_cats = 0
        self.user_side_mat = user_side_mat

    def get_item_side_mat(self):
        try:
            item_side_mat = sp.load_npz(self.item_side_file)
            self.n_i_cats = item_side_mat.shape[1]
            print('already load item side mat', item_side_mat.shape)
        except Exception:
            item_side_mat = None
            self.n_i_cats = 0
        self.item_side_mat = item_side_mat

    def get_adj_mat(self):
        try:
            t1 = time()
            bi_adj_mat = sp.load_npz(self.path + '/s_bi_adj_mat.npz')
            si_adj_mat = sp.load_npz(self.path + '/s_si_adj_mat.npz')
            print('already load adj matrix', bi_adj_mat.shape, time() - t1)
        except Exception:
            bi_adj_mat, si_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_bi_adj_mat.npz', bi_adj_mat)
            sp.save_npz(self.path + '/s_si_adj_mat.npz', si_adj_mat)

        return bi_adj_mat, si_adj_mat

    def create_adj_mat(self):
        print('Creating adj mat ...')
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items + self.n_u_cats + self.n_i_cats,
                                 self.n_users + self.n_items + self.n_u_cats + self.n_i_cats), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        adj_mat[:self.n_users, self.n_users: self.n_users + self.n_items] = self.sp_train.tolil()
        adj_mat[self.n_users: self.n_users + self.n_items, :self.n_users] = self.sp_train.tolil().T

        if self.user_social_mat is not None:
            adj_mat[:self.n_users, : self.n_users] = self.user_social_mat.tolil()
            print('\tincluding user social relations.')

        if self.user_side_mat is not None:
            adj_mat[:self.n_users, self.n_users + self.n_items: self.n_users + self.n_items + self.n_u_cats] \
                = self.user_side_mat.tolil()
            adj_mat[self.n_users + self.n_items: self.n_users + self.n_items + self.n_u_cats, :self.n_users] \
                = self.user_side_mat.tolil().T
            print('\tincluding user side information.')

        if self.item_side_mat is not None:
            adj_mat[self.n_users: self.n_users + self.n_items, self.n_users + self.n_items + self.n_u_cats:] \
                = self.item_side_mat.tolil()
            adj_mat[self.n_users + self.n_items + self.n_u_cats:, self.n_users: self.n_users + self.n_items] \
                = self.item_side_mat.tolil().T
            print('\tincluding item side information.')

        adj_mat = adj_mat.todok()
        print('alread create adjacency matrix', adj_mat.shape, time()-t1)

        t2 = time()

        def normalized_adj_bi(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            print('\tgenerate bi-normalized adjacency matrix.')
            return bi_adj

        def normalized_adj_si(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('\tgenerate si-normalized adjacency matrix.')
            return norm_adj

        bi_adj_mat = normalized_adj_bi(adj_mat + sp.eye(adj_mat.shape[0]))
        si_adj_mat = normalized_adj_si(adj_mat + sp.eye(adj_mat.shape[0]))

        print('already normalize adjacency matrix', time() - t2)
        return bi_adj_mat, si_adj_mat

    def generate_sp_train_batch(self, batch_size):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items-1, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in self.test_set[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        instance_idx = []
        while True:
            if len(instance_idx) == batch_size: break

            idx = np.random.randint(low=0, high=len(self.sp_train.row), size=1)[0]
            instance_idx.append(idx)
        # get the indices of users, positive items, and negative items.
        user_idx = self.sp_train.row[instance_idx]
        pos_item_idx = self.sp_train.col[instance_idx]

        #
        # # ---------check------------
        # for c_id, u_id in enumerate(user_idx):
        #     iid = pos_item_idx[c_id]
        #     if iid not in self.train_items[u_id]:
        #         print('----check error---')
        #         exit()
        # # ---------check end------------


        neg_item_idx = []

        for u in user_idx:
            neg_item_idx += sample_neg_items_for_u(u, 1)
        neg_item_idx = np.array(neg_item_idx)

        users = self.user_one_hot[user_idx]
        # Convert from indices to one-hot matrices
        if self.user_social_mat is not None:
            # Add the social network to the users' one-hot representation.
            user_social = self.user_social_mat[user_idx] + users
        else:
            user_social = users
        pos_items = self.item_one_hot[pos_item_idx]
        neg_items = self.item_one_hot[neg_item_idx]

        # Horizontally stack sparse matrices to get single positive
        # and negative feature matrices
        pos_feats = sp.hstack([user_social, pos_items])
        neg_feats = sp.hstack([user_social, neg_items])

        if self.user_side_mat is not None:
            user_side = self.user_side_mat[user_idx]
            pos_feats = sp.hstack([pos_feats, user_side])
            neg_feats = sp.hstack([neg_feats, user_side])

        if self.item_side_mat is not None:
            pos_item_side = self.item_side_mat[pos_item_idx]
            neg_item_side = self.item_side_mat[neg_item_idx]
            pos_feats = sp.hstack([pos_feats, pos_item_side])
            neg_feats = sp.hstack([neg_feats, neg_item_side])

        return (users, pos_items, neg_items, pos_feats, neg_feats)

    def get_n_features(self):
        n_features = self.n_users + self.n_items

        if self.user_side_mat is not None:
            n_user_side = self.user_side_mat.shape[1]
        else:
            n_user_side = 0

        if self.item_side_mat is not None:
            n_item_side = self.item_side_mat.shape[1]
        else:
            n_item_side = 0

        print('[n_org_features, n_user_side, n_item_side]=[%d, %d, %d]' % (n_features, n_user_side, n_item_side))
        print('[n_train, n_test, n_test_neg]=[%d, %d, %d]' % (len(self.sp_train.row), len(self.sp_test.row),
                                                              len(self.sp_test_neg.row)))

        self.n_features = n_features + n_user_side + n_item_side