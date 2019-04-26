#!/usr/local/bin/bash
import utility.metrics as metrics
import multiprocessing
import heapq
import numpy as np
import scipy.sparse as sp

class Tester(object):
    def __init__(self, args, data_generator):
        self.Ks = eval(args.Ks)

        self.path = args.data_path
        self.dataset = args.dataset

        self.batch_size = args.batch_size
        self.test_flag = args.test_flag

        self.layer_size = args.layer_size

        self.data_generator = data_generator
        self.n_features = data_generator.n_features
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items

        self.n_train = data_generator.n_train
        self.n_test = data_generator.n_test

        self.cores = multiprocessing.cpu_count()

    def pack_batch_feats(self, _u_batch, _i_batch):
        u_batch = np.repeat(_u_batch, len(_i_batch), axis=0).tolist()
        i_batch = list(_i_batch) * len(_u_batch)

        assert len(u_batch) == len(i_batch)

        # Horizontally stack sparse matrices to get single feature matrices w.r.t. user-item interactions.
        if self.data_generator.user_social_mat is not None:
            users = self.data_generator.user_social_mat[u_batch] + self.data_generator.user_one_hot[u_batch]
        else:
            users = self.data_generator.user_one_hot[u_batch]

        items = self.data_generator.item_one_hot[i_batch]
        feat_batch = sp.hstack([users, items])

        # Horizontally stack sparse matrices to get single feature matrics w.r.t. user & item side info.
        if self.data_generator.user_side_mat is not None:
            user_side = self.data_generator.user_side_mat[u_batch]
            feat_batch = sp.hstack([feat_batch, user_side])

        if self.data_generator.item_side_mat is not None:
            item_side = self.data_generator.item_side_mat[i_batch]
            feat_batch = sp.hstack([feat_batch, item_side])

        return (users, feat_batch)

    def create_feed_dict(self, placeholder, users, feats):
        feed_dict = {
            placeholder['user_list']: users.nonzero()[1],
            placeholder['pos_indices']: np.hstack((feats.nonzero()[0][:, None], feats.nonzero()[1][:, None])),
            placeholder['pos_values']: feats.data,
            placeholder['pos_shape']: feats.shape,
        }
        return feed_dict

    def test(self, sess, model, users_to_test, drop_flag=False, batch_test_flag=True):
        pool = multiprocessing.Pool(self.cores)

        result = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)), 'ndcg': np.zeros(len(self.Ks)),
                  'hit_ratio': np.zeros(len(self.Ks)), 'auc': 0.}

        u_batch_size = self.batch_size
        i_batch_size = self.batch_size
        # u_batch_size = 10
        # i_batch_size = 10

        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            if batch_test_flag:
                n_item_batchs = self.n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), self.n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, self.n_items)

                    item_batch = range(i_start, i_end)

                    (user_list, feat_batch) = self.pack_batch_feats(user_batch, item_batch)

                    feed_dict = {model.user_list: user_list.nonzero()[1],

                                 model.pos_indices: np.hstack(
                                     (feat_batch.nonzero()[0][:, None], feat_batch.nonzero()[1][:, None])),
                                 model.pos_values: feat_batch.data,
                                 model.pos_shape: feat_batch.shape,
                                 }

                    if drop_flag == False:

                        i_rate_batch = sess.run(model.batch_predictions, feed_dict=feed_dict)
                    else:
                        i_rate_batch = sess.run(model.batch_predictions, feed_dict=feed_dict)

                    rate_batch[:, i_start: i_end] = i_rate_batch.reshape((-1, i_end - i_start))
                    i_count += i_end - i_start

                    # print('%d over %d; %d over %d.' % (u_batch_id, n_user_batchs, i_batch_id, n_item_batchs))
                assert i_count == self.n_items

            else:
                item_batch = range(self.n_items)
                (user_list, feat_batch) = self.pack_batch_feats(user_batch, item_batch)
                feed_dict = {model.user_list: user_list.nonzero()[1],

                             model.pos_indices: np.hstack(
                                 (feat_batch.nonzero()[0][:, None], feat_batch.nonzero()[1][:, None])),
                             model.pos_values: feat_batch.data,
                             model.pos_shape: feat_batch.shape,
                             }

                if drop_flag == False:
                    rate_batch = sess.run(model.batch_predictions, feed_dict=feed_dict)
                else:
                    rate_batch = sess.run(model.batch_predictions, feed_dict=feed_dict)

            user_batch_rating_uid = zip(rate_batch, user_batch)
            batch_result = pool.map(self.test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result

    def test_one_user(self, x):
        # user u's ratings for user u
        rating = x[0]
        # uid
        u = x[1]
        # user u's items in the training set
        try:
            training_items = self.data_generator.train_items[u]
        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = self.data_generator.test_set[u]

        all_items = set(range(self.n_items))

        test_items = list(all_items - set(training_items))

        if self.test_flag == 'part':
            r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)
            # print(r, auc)
        else:
            r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating, self.Ks)

        return self.get_performance(user_pos_test, r, auc, self.Ks)

    def ranklist_by_heapq(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        # calculate the auc
        auc = 0.
        # auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    def get_auc(self, item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = metrics.auc(ground_truth=r, prediction=posterior)
        return auc

    def ranklist_by_sorted(self, user_pos_test, test_items, rating, Ks):
        item_score = []
        for i in test_items:
            item_score.append((i, rating[i]))

        item_score = sorted(item_score, key=lambda x: x[1])
        item_score.reverse()

        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = metrics.auc(ground_truth=r, prediction=posterior)
        return r, auc

    def get_performance(self, user_pos_test, r, auc, Ks):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in Ks:
            # precision.append(metrics.precision_at_k(r, K))
            precision.append(metrics.average_precision(r, K))
            recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r, K))
            hit_ratio.append(metrics.hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}







