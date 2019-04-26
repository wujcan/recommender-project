#!/usr/local/bin/bash
import multiprocessing
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import pandas as pd


class Tester(object):
    def __init__(self, args, path, data_generator):
        self.path = path

        self.model_type = args.model_type

        self.batch_size = args.batch_size
        self.test_flag = args.test_flag

        self.data_generator = data_generator
        self.n_features = data_generator.n_features
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items

        self.n_train = data_generator.n_train
        self.n_valid = len(data_generator.sp_valid.row)
        self.n_test = len(data_generator.sp_test.row)

        self.sp_valid = data_generator.sp_valid
        self.feat_valid = data_generator.valid_feats

        self.sp_test = data_generator.sp_test
        self.feat_test = data_generator.test_feats

        self.cores = multiprocessing.cpu_count()

    def pack_batch_feats(self, u_batch, i_batch, feats_batch):
        # Horizontally stack sparse matrices to get single feature matrices w.r.t. user-item interactions.
        users = self.data_generator.user_one_hot[u_batch]

        items = self.data_generator.item_one_hot[i_batch]
        feat_batch = sp.hstack([items, feats_batch])

        return (users, feat_batch)

    def create_feed_dict(self, placeholder, users, feats):
        feed_dict = {
            placeholder['user_list']: users.nonzero()[1],
            placeholder['pos_indices']: np.hstack((feats.nonzero()[0][:, None], feats.nonzero()[1][:, None])),
            placeholder['pos_values']: feats.data,
            placeholder['pos_shape']: feats.shape,
        }
        return feed_dict

    def test(self, sess, model, drop_flag=False, phase='Validation'):
        result = {'WF1': 0.}

        def test_part(insts, feats):
            inst_batch_size = self.batch_size
            n_insts = len(insts.row)
            n_inst_batchs = n_insts // inst_batch_size + 1

            preds = np.zeros(shape=(n_insts))

            for inst_batch_id in range(n_inst_batchs):
                start = inst_batch_id * inst_batch_size
                end = min((inst_batch_id + 1) * inst_batch_size, n_insts)

                idx = list(range(start, end))

                user_batch = insts.row[idx]
                item_batch = insts.col[idx]
                feat = []
                for row in idx:
                    feat_mat = feats.getrow(row)
                    if feat == []:
                        feat = feat_mat
                    else:
                        feat = sp.vstack([feat, feat_mat])

                (user_list, feat_batch) = self.pack_batch_feats(user_batch, item_batch, feat)

                feed_dict = {model.user_list: user_list.nonzero()[1],
                             model.sp_indices: np.hstack(
                                 (feat_batch.nonzero()[0][:, None], feat_batch.nonzero()[1][:, None])),
                             model.sp_values: feat_batch.data,
                             model.sp_shape: feat_batch.shape,
                             }

                if drop_flag == False:
                    pred_batch = sess.run(model.batch_predictions, feed_dict=feed_dict)
                else:
                    pred_batch = sess.run(model.batch_predictions, feed_dict=feed_dict)
                index = user_batch
                preds[start: end] = np.array(pred_batch).reshape(-1)
                for i in range(len(index)):
                    dictindex[str(index[i])].append(preds[i])

        if phase == 'Validation':
            index2sid = self.path + "/index_to_sid_validset.data"
            index2mode = self.path + "/valid.data"
        else:
            index2sid = self.path + "/index_to_sid_test.data"
            index2mode = self.path + "/test.data"
            savefile = self.path + "/predict.csv"

        findex2sid = open(index2sid)
        findex2mode = open(index2mode)
        dictindex = {}
        for line in findex2sid:
            line = line.split()
            dictindex[line[0]] = [line[1]]
        for line in findex2mode:
            line = line.split(',')
            dictindex[line[0]].append(line[1])
            dictindex[line[0]].append(line[2])

        if phase == 'Validation':
            test_part(self.sp_valid, self.feat_valid)
        else:
            test_part(self.sp_test, self.feat_test)

        dictind = [item[1] for item in list(dictindex.items())]
        df = pd.DataFrame(data=dictind)
        df.columns = ['sid', 'label', 'transport_mode', 'predict']

        if phase == 'Validation':
            df['label'] = df['label'].astype('int')
            df['transport_mode'] = df['transport_mode'].astype('int')
            df['transport_mode'] -= 1
            df['click_mode'] = df['label'] * df['transport_mode']
            # find click mode for each sid
            res1_idx = df.groupby(['sid'], sort=False)['click_mode'].transform(max) == df['click_mode']
            df_click_mode = df[res1_idx]
            df_click_mode.drop_duplicates('sid', inplace=True)
            # get predict mode
            result_idx = df.groupby(['sid'], sort=False)['predict'].transform(max) == df['predict']
            df_result = df[result_idx]
            df_result.drop_duplicates('sid', inplace=True)
            df_result.rename(columns={'transport_mode': 'recommend_mode'}, inplace=True)
            df_predict = df_result[['sid', 'recommend_mode']]
            df_click = df_click_mode[['sid', 'click_mode']]
            df_final = pd.merge(df_click, df_predict, how='left')
            score = f1_score(df_final['recommend_mode'], df_final['click_mode'], average='weighted')
            result['WF1'] = score

            return result

        else:
            # get recommend mode
            result_idx = df.groupby(['sid'], sort=False)['predict'].transform(max) == df['predict']
            df_result = df[result_idx]
            df_result.rename(columns={'transport_mode': 'recommend_mode'}, inplace=True)
            df_final = df_result[['sid', 'recommend_mode']]
            df_final.to_csv(savefile, index=False)
