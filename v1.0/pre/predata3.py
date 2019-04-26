#!/usr/local/bin/bash
import pandas as pd
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import copy
import os

class PreData(object):
    def __init__(self, user_dict, item_dict, path, user_social_dict=None,
                 user_side_dict=None, item_side_dict=None, n_u_f=5, n_i_f=5):
        # ----------------------------- deal with user-item interactions ---------------------------
        # store user-items, item-users interactions.
        self.user_dict, self.item_dict = user_dict, item_dict

        # filter inactive users & items out.
        self.n_u_f, self.n_i_f = n_u_f, n_i_f

        # remap the original ids to the new ones.
        self.user_list, self.item_list = [], []
        self.remap_user_dict = {}

        # split the ratings into training & testing; then store.
        self.train_ratio = 0.8
        self.train_path = '%s/train.dat' % path
        self.test_path = '%s/test.dat' % path

        self.test_neg_path = '%s/test_neg.dat' % path

        # store remap lists.
        self.user_list_path = '%s/user_list.dat' % path
        self.item_list_path = '%s/item_list.dat' % path

        self.interaction_filter()
        self.interaction_split()

        # ----------------------------- deal with user-item interactions ---------------------------
        # store user-side, item-side contextual information.
        self.user_social_dict, self.user_side_dict, self.item_side_dict = user_social_dict, user_side_dict, item_side_dict

        self.user_social_path = '%s/user_social.npz' % path
        self.user_side_path = '%s/user_side.npz' % path
        self.item_side_path = '%s/item_side.npz' % path

        self.u_cat_list, self.i_cat_list = [], []
        self.u_cat_list_path = '%s/u_cat_list.dat' % path
        self.i_cat_list_path = '%s/i_cat_list.dat' % path

        self.context_load()
        self.store()


    def interaction_filter(self):
        f_user_dict, f_item_dict = {}, {}

        print('n_users\tn_items')
        while True:
            print(len(self.user_dict.keys()), len(self.item_dict.keys()))
            flag1, flag2 = True, True

            for u_id in self.user_dict.keys():
                pos_items = self.user_dict[u_id]
                val_items = [idx for idx in pos_items if idx in self.item_dict.keys()]

                if len(val_items) >= self.n_u_f:
                    f_user_dict[u_id] = val_items
                else:
                    flag1 = False

            self.user_dict = f_user_dict.copy()

            for i_id in self.item_dict.keys():
                pos_users = self.item_dict[i_id]
                val_users = [udx for udx in pos_users if udx in self.user_dict.keys()]

                if len(pos_users) >= self.n_i_f:
                    f_item_dict[i_id] = val_users
                else:
                    flag2 = False

            self.item_dict = f_item_dict.copy()
            f_user_dict, f_item_dict = {}, {}

            if flag1 and flag2:
                print('filter done.')
                break

    def interaction_split(self):
        user_dict = self.user_dict.copy()
        n_rates = 0

        f1 = open(self.train_path, 'w')
        f2 = open(self.test_path, 'w')

        # get the training rates.
        for u_id in user_dict.keys():
            n_items = len(user_dict[u_id])
            n_rates += n_items

            # split the rating of each user into [0:n_train] & [n_train:]
            if n_items <= 2:
                n_train = 2
            else:
                n_train = int(n_items * self.train_ratio)

            # remap the user id.
            if u_id not in self.user_list:
                self.user_list.append(u_id)
            new_u_id = self.user_list.index(u_id)
            new_i_ids = []

            # remap the item id.
            for i_id in user_dict[u_id][0:n_train]:
                if i_id not in self.item_list:
                    self.item_list.append(i_id)
                new_i_id = self.item_list.index(i_id)
                new_i_ids.append(new_i_id)
            f1.write(str(new_u_id) + ' ' + ' '.join([str(i) for i in new_i_ids]) + '\n')

        # get the testing rates.
        for u_id in user_dict.keys():
            n_items = len(user_dict[u_id])

            if n_items <= 2:
                n_train = 2
            else:
                n_train = int(n_items * self.train_ratio)

            new_u_id = self.user_list.index(u_id)
            new_i_ids = []

            for i_id in user_dict[u_id][n_train:]:
                if i_id in self.item_list:
                    new_i_id = self.item_list.index(i_id)
                    new_i_ids.append(new_i_id)
            f2.write(str(new_u_id) + ' ' + ' '.join([str(i) for i in new_i_ids]) + '\n')

        f1.close()
        f2.close()
        print('n_users=%d, n_items=%d' % (len(self.user_list), len(self.item_list)))
        print('store %d ratings done.' % n_rates)
        print('\ttrain ratings in ', self.train_path)
        print('\ttest raitngs in ', self.test_path)

    def store(self):
        user_list = self.user_list.copy()
        item_list = self.item_list.copy()
        u_cat_list = self.u_cat_list.copy()
        i_cat_list = self.i_cat_list.copy()

        f1 = open(self.user_list_path, 'w')
        f2 = open(self.item_list_path, 'w')
        f3 = open(self.u_cat_list_path, 'w')
        f4 = open(self.i_cat_list_path, 'w')

        f1.write('org_id remap_id\n')
        for i, org in enumerate(user_list):
            f1.write(str(org) + ' ' + str(i) + '\n')

        f2.write('org_id remap_id\n')
        for i, org in enumerate(item_list):
            f2.write(str(org) + ' ' + str(i) + '\n')

        f3.write('org_id remap_id\n')
        for i, org in enumerate(u_cat_list):
            f3.write(str(org) + ' ' + str(i) + '\n')

        f4.write('org_id remap_id\n')
        for i, org in enumerate(i_cat_list):
            f4.write(str(org) + ' ' + str(i) + '\n')

        f1.close()
        f2.close()
        f3.close()
        f4.close()

    def context_load(self):
        if self.user_social_dict is not None:
            self.get_user_social_mat()

        if self.user_side_dict is not None:
            self.get_user_side_mat()

        if self.item_side_dict is not None:
            self.get_item_side_mat()


    def get_user_social_mat(self):
        cat_rows = []
        cat_cols = []
        cat_data = []

        for u_id in self.user_social_dict.keys():
            if u_id not in self.user_list:
                continue
            new_u_id = self.user_list.index(u_id)

            for f_id in self.user_social_dict[u_id]:
                if f_id not in self.user_list:
                    continue
                new_f_id = self.user_list.index(f_id)

                cat_rows.append(new_u_id)
                cat_cols.append(new_f_id)
                cat_data.append(1)
        n_users = len(self.user_list)
        user_social_mat = sp.coo_matrix((cat_data, (cat_rows, cat_cols)), shape=(n_users, n_users)).tocsr()

        sp.save_npz(self.user_social_path, user_social_mat)
        print('\tuser social info save:', self.user_social_path)
        return

    def get_user_side_mat(self):
        user_side_mat = self.get_side_mat(tag='user')
        sp.save_npz(self.user_side_path, user_side_mat)
        print('\tuser side info save:', self.user_side_path)
        return

    def get_item_side_mat(self):
        item_side_mat = self.get_side_mat(tag='item')
        sp.save_npz(self.item_side_path, item_side_mat)
        print('\titem side info save:', self.item_side_path)
        return

    def get_side_mat(self, tag):
        if tag == 'user':
            side_dict = self.user_side_dict
            cat_list = self.u_cat_list
            idx_list = self.user_list
        else:
            side_dict = self.item_side_dict
            cat_list = self.i_cat_list
            idx_list = self.item_list

        cat_rows = []
        cat_cols = []
        cat_data = []

        for org_id in side_dict.keys():
            if org_id not in idx_list:
                continue
            new_id = idx_list.index(org_id)

            for cat in side_dict[org_id]:
                temps = cat.strip().split(':')
                c_name = temps[0]
                c_val = float(temps[1])

                if c_name not in cat_list:
                    cat_list.append(c_name)
                c_id = cat_list.index(c_name)

                cat_rows.append(new_id)
                cat_cols.append(c_id)
                cat_data.append(c_val)

        side_mat = sp.coo_matrix((cat_data, (cat_rows, cat_cols))).tocsr()
        return side_mat




