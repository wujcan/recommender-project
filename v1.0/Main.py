#!/usr/local/bin/bash
import numpy as np
import tensorflow as tf
from utility.load_data import Data
from utility.helper import *
from utility.parser import parse_args
from utility.batch_test import Tester
from time import time
from FM import FM
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_pretrained_data(args):
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, args.model_type)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def genSparseTensor(feats):
    sp_indices = np.hstack((feats.nonzero()[0][:, None], feats.nonzero()[1][:, None]))
    sp_values = feats.data
    sp_shape = feats.shape
    return sp_indices, sp_values, sp_shape


if __name__ == '__main__':
    np.random.seed(2019)
    # get argument settings.
    args = parse_args()

    # get data generator.
    data_generator = Data(path=args.proj_path + args.data_path + args.dataset)

    config = dict()
    config['n_features'] = data_generator.n_features
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    # use pretrained parameters from the basic model; note that it is not fine-tune.
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    # *********************************************************
    # init FM model
    model = FM(data_config=config, pretrain_data=pretrain_data, args=args)
    saver = tf.train.Saver()

    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                         '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    tester = Tester(args=args, data_generator=data_generator, path=args.proj_path + args.data_path + args.dataset)

    # *********************************************************
    # reload the model parameters to fine tune.
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/l%s_r%s' % \
                        (args.proj_path, args.dataset, model.model_type, str(args.lr),
                         '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from the model to fine tune.
            if args.report != 1:
                ret = tester.test(sess, model, drop_flag=False)
                cur_best_WF1 = ret['WF1']

                pretrain_ret = 'pretrained model WF1=[%.5f]' % \
                               (ret['WF1'])

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
            cur_best_WF1 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_WF1 = 0.
        print('without pretraining.')

    # get the final report, as well as the performance w.r.t. different sparsity.
    if args.report == 1:

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        ret = tester.test(sess, model, drop_flag=False)

        final_perf = "WF1=[%.5f]" % \
                     (ret['WF1'])
        print(final_perf)

        f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    loss_loger,  WF1_loger = [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, log_loss, reg_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        t11 = time()
        for idx in range(n_batch):
            (items, feats, sp_labels) = data_generator.generate_sp_train_batch(args.batch_size)
            sp_indices, sp_values, sp_shape = genSparseTensor(feats)

            feed_dict = {model.sp_indices: sp_indices,
                         model.sp_values: sp_values,
                         model.sp_shape: sp_shape,
                         model.sp_labels: sp_labels
                         }

            _, batch_loss, batch_log_loss, batch_reg_loss, batch_preds = sess.run(
                [model.opt, model.loss, model.log_loss, model.reg_loss, model.preds],
                feed_dict=feed_dict
            )

            loss += batch_loss / n_batch
            log_loss += batch_log_loss / n_batch
            reg_loss += batch_reg_loss / n_batch

        print('time for training each batch [%.1f s].' % ((time() - t1) / n_batch))

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if args.dataset == 'debug':
            test_intetval = 5
        else:
            test_intetval = 10
        # print the test evaluation metrics each 10 epochs.
        if (epoch + 1) % test_intetval != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, log_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()

        ret = tester.test(sess, model, drop_flag=False, phase='Validation')

        t3 = time()

        loss_loger.append(loss)
        WF1_loger.append(ret['WF1'])

        if cur_best_WF1 < ret['WF1']:
            tester.test(sess, model, drop_flag=False, phase='Test')

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], WF1=[%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, log_loss, reg_loss, ret['WF1'])
            print(perf_str)

        cur_best_WF1, stopping_step, should_stop = early_stopping(ret['WF1'], cur_best_WF1,
                                                                  stopping_step, expected_order='acc',
                                                                  flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['WF1'] == cur_best_WF1 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    WF1 = np.array(WF1_loger)

    best_WF1 = max(WF1)
    idx = list(WF1).index(best_WF1)

    final_perf = "Best Iter=[%d]@[%.1f]\tWF1=[%.5f]" % \
                 (idx, time() - t0,  WF1[idx])
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n\t%s\n' % (args.embed_size, args.lr, args.regs,
                                                                         args.loss_type, final_perf))
    f.close()
