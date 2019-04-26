#!/usr/local/bin/bash
import numpy as np
import tensorflow as tf
from utility.load_data3 import Data
from utility.helper import *
from utility.parser import parse_args
from utility.batch_test5 import Tester
import utility.metrics as metrics
from time import time
import scipy.sparse as sp
from FM_pro import FM
from MF import MF
from DEPFM import DEPFM
from TransFM import TransFM
from TransDEPFM import TransDEPFM


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_pretrained_data(args):
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'fm')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data

def create_feed_dict(placeholder, users, feats):
    feed_dict = {
        placeholder['user_list']: users.nonzero()[1],
        placeholder['pos_indices']: np.hstack((feats.nonzero()[0][:, None], feats.nonzero()[1][:, None])),
        placeholder['pos_values']: feats.data,
        placeholder['pos_shape']: feats.shape,
    }
    return feed_dict

def merge2feats(pos_feats, neg_feats):
    all_feats = sp.vstack((pos_feats, neg_feats))

    sp_indices = np.hstack((all_feats.nonzero()[0][:, None], all_feats.nonzero()[1][:, None]))
    sp_values = np.concatenate((pos_feats.data, neg_feats.data))
    sp_shape = all_feats.shape

    sp_labels = np.concatenate((np.ones(shape=(pos_feats.shape[0])),
                                np.zeros(shape=(neg_feats.shape[0]))), axis=0)

    return sp_indices, sp_values, sp_shape, sp_labels

if __name__ == '__main__':
    np.random.seed(2019)
    # get argument settings.
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # get data generator.
    data_generator = Data(path=args.data_path + args.dataset)

    config = dict()
    config['n_features'] = data_generator.n_features
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    if args.model_type in ['depfm', 'transdepfm']:
        bi_adj, si_adj = data_generator.get_adj_mat()
        if args.adj_type == 'bi':
            config['norm_adj'] = bi_adj
            print('use the bilinear adjacency matrix')

        elif args.adj_type == 'si':
            config['norm_adj'] = si_adj
            print('use the single adjacency matrix')

    t0 = time()

    # use pretrained parameters from the basic model; note that it is not fine-tune.
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    # *********************************************************
    # select one of the models.
    if args.model_type == 'fm':
        model = FM(data_config=config, pretrain_data=pretrain_data, args=args)
    elif args.model_type == 'mf':
        model = MF(data_config=config, pretrain_data=pretrain_data, args=args)
    elif args.model_type == 'depfm':
        model = DEPFM(data_config=config, pretrain_data=pretrain_data, args=args)
    elif args.model_type == 'transfm':
        model = TransFM(data_config=config, pretrain_data=pretrain_data, args=args)
    elif args.model_type == 'transdepfm':
        model = TransDEPFM(data_config=config, pretrain_data=pretrain_data, args=args)

    saver = tf.train.Saver()

    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        if args.model_type in ['mf', 'fm']:
            weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                             '-'.join([str(r) for r in eval(args.regs)]))
        elif args.model_type in ['depfm', 'nfm', 'transfm', 'deepfm', 'transdepfm']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    tester = Tester(args=args, data_generator=data_generator)

    # *********************************************************
    # reload the model parameters to fine tune.
    if args.pretrain == 1:
        if args.model_type in ['mf', 'fm']:
            pretrain_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                             '-'.join([str(r) for r in eval(args.regs)]))
        elif args.model_type in ['depfm', 'nfm', 'transfm', 'deepfm', 'transdepfm']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from the model to fine tune.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())

                ret = tester.test(sess, model, users_to_test, drop_flag=False)
                cur_best_auc = ret['auc']

                pretrain_ret = 'pretrained model logloss=[%.5f], auc=[%.5f]' % \
                               (ret['logloss'], ret['auc'])

                print(pretrain_ret)

                # *********************************************************
                # save the pretrained model parameters for pretraining other models.
                if args.save_flag == -1:
                    var_linear, var_factor = sess.run(
                        [model.weights['var_linear'], model.weights['var_factor']],
                        feed_dict={})
                    # temp_save_path = '%spretrain/%s/%s/%s_%s.npz' % (args.proj_path, args.dataset, args.model_type, str(args.lr),
                    #                                                  '-'.join([str(r) for r in eval(args.regs)]))
                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, model.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, var_linear=var_linear, var_factor=var_factor)
                    print('save the weights of fm in path: ', temp_save_path)
                    exit()

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_auc = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_auc = 0.
        print('without pretraining.')

    # get the final report, as well as the performance w.r.t. different sparsity.
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()

        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = tester.test(sess, model, users_to_test, drop_flag=False)

            final_perf = "logloss=[%s], auc=[%.5f]" % \
                         (ret['logloss'], ret['auc'])
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    loss_loger, logloss_loger, auc_loger = [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, log_loss, reg_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            btime= time()
            (users, pos_items, neg_items, pos_feats, neg_feats) = data_generator.generate_sp_train_batch(args.batch_size)

            sp_indices, sp_values, sp_shape, sp_labels = merge2feats(pos_feats, neg_feats)

            if args.model_type in ['fm', 'depfm', 'transfm', 'nfm', 'deepfm', 'transdepfm']:
                feed_dict = {model.user_list: users.nonzero()[1],

                             model.sp_indices: sp_indices,
                             model.sp_values: sp_values,
                             model.sp_shape: sp_shape,
                             model.sp_labels: sp_labels
                             }
            elif args.model_type in ['mf']:
                feed_dict = {model.user_list: np.concatenate((users.nonzero()[1], users.nonzero()[1])),
                             model.item_list: np.concatenate((pos_items.nonzero()[1], neg_items.nonzero()[1])),

                             model.sp_labels: sp_labels
                             }

            _, batch_loss, batch_log_loss, batch_reg_loss, batch_preds = sess.run(
                [model.opt, model.loss, model.log_loss, model.reg_loss, model.preds],
                feed_dict=feed_dict
            )

            loss += batch_loss / n_batch
            log_loss += batch_log_loss / n_batch
            reg_loss += batch_reg_loss / n_batch
            # print(time() - btime, batch_loss, batch_log_loss, batch_reg_loss)
            #
            # print('\n')
            # print(batch_preds)
            # print(metrics.auc(sp_labels, batch_preds))
            # exit()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, log_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = tester.test(sess, model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        logloss_loger.append(ret['logloss'])
        auc_loger.append(ret['auc'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], logloss=[%.5f], auc=[%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, log_loss, reg_loss, ret['logloss'], ret['auc'])
            print(perf_str)

        cur_best_auc, stopping_step, should_stop = early_stopping(ret['auc'], cur_best_auc,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['auc'] == cur_best_auc and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)


    logloss = np.array(logloss_loger)
    auc = np.array(auc_loger)

    best_auc = max(auc)
    idx = list(auc).index(best_auc)

    final_perf = "Best Iter=[%d]@[%.1f]\tlogloss=[%s], auc=[%.5f]" % \
                 (idx, time() - t0, logloss[idx], auc[idx])
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n\t%s\n' % (args.embed_size, args.lr, args.regs,
                                                                         args.loss_type, final_perf))
    f.close()








