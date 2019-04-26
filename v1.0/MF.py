#!/usr/local/bin/bash
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MF(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'mf'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.regs = eval(args.regs)

        self.verbose = args.verbose

        # placeholder definition
        self.user_list = tf.placeholder(tf.int32, shape=[None,], name='users')
        self.item_list = tf.placeholder(tf.int32, shape=[None,], name='items')

        self.sp_labels = tf.placeholder(tf.float32, shape=[None], name='sp_labels')

        self.weights = self._init_weights()


        # All predictions for all users.
        self.batch_predictions = self._create_mf_interactions()

        self.log_loss, self.reg_loss = self._create_log_loss()

        self.loss = self.log_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                    name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')
        return all_weights

    def _create_log_loss(self):
        self.preds = self._create_mf_interactions()

        log_loss = tf.reduce_mean(tf.losses.log_loss(labels=tf.reshape(self.sp_labels, [-1]),
                                                     predictions=tf.reshape(self.preds, [-1])))

        reg_loss = self.regs[0] * (tf.nn.l2_loss(self.weights['user_embedding']) +
                                   tf.nn.l2_loss(self.weights['item_embedding']))

        return log_loss, reg_loss

    def _create_mf_interactions(self):
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_list)
        i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_list)

        predictions = tf.sigmoid(tf.reduce_sum(tf.multiply(u_e, i_e), axis=1))

        return predictions


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)