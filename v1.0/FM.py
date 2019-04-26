#!/usr/local/bin/bash
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class FM(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'fm'
        self.pretrain_data = pretrain_data

        self.n_features = data_config['n_features']

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.regs = eval(args.regs)

        self.verbose = args.verbose

        # Sparse placeholder definition
        self.user_list = tf.placeholder(tf.int64, shape=[None], name='user_list')

        self.sp_indices = tf.placeholder(tf.int64, shape=[None, 2], name='sp_indices')
        self.sp_values = tf.placeholder(tf.float32, shape=[None], name='sp_values')
        self.sp_shape = tf.placeholder(tf.int64, shape=[2], name='sp_shape')
        self.sp_labels = tf.placeholder(tf.float32, shape=[None], name='sp_labels')

        # Input positive features, shape=(batch_size * feature_dim)
        sp_feats = tf.SparseTensor(self.sp_indices, self.sp_values, self.sp_shape)

        self.weights = self._init_weights()

        # All predictions for all users.
        self.batch_predictions = self._create_bi_predictions(sp_feats)

        self.log_loss, self.reg_loss = self._create_log_loss(sp_feats)

        self.loss = self.log_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['var_linear'] = tf.Variable(initializer([self.n_features, 1]), name='var_linear')
            all_weights['var_factor'] = tf.Variable(initializer([self.n_features, self.emb_dim]), name='var_factor')
            print('using xavier initialization')
        else:
            all_weights['var_linear'] = tf.Variable(initial_value=self.pretrain_data['var_linear'], trainable=True,
                                                    name='var_linear', dtype=tf.float32)
            all_weights['var_factor'] = tf.Variable(initial_value=self.pretrain_data['var_factor'], trainable=True,
                                                    name='var_factor', dtype=tf.float32)
            print('using pretrained initialization')
        return all_weights

    def _create_log_loss(self, sp_feats):
        self.preds = self._create_bi_predictions(sp_feats)

        log_loss = tf.reduce_mean(tf.losses.log_loss(labels=tf.reshape(self.sp_labels, [-1]),
                                      predictions=tf.reshape(self.preds, [-1])))

        reg_loss = self.regs[0] * tf.nn.l2_loss(self.weights['var_linear']) + \
                      self.regs[1] * tf.nn.l2_loss(self.weights['var_factor'])

        return log_loss, reg_loss

    def _create_bi_predictions(self, feats):
        # Linear terms.
        term0 = tf.sparse_tensor_dense_matmul(feats, self.weights['var_linear'])
        # Interaction terms w.r.t. first sum then square.
        #   e.g., sum_{k from 1 to K}{(v1k+v2k)**2}
        sum_emb = tf.sparse_tensor_dense_matmul(feats, self.weights['var_factor'])
        term1 = tf.square(tf.reduce_sum(sum_emb, axis=1, keepdims=True))

        # Interaction terms w.r.t. first square then sum.
        #   e.g., sum_{k from 1 to K}{v1k**2 + v2k**2}
        square_emb = tf.sparse_tensor_dense_matmul(tf.square(feats), tf.square(self.weights['var_factor']))
        term2 = tf.reduce_sum(square_emb, axis=1, keepdims=True)

        preds = tf.sigmoid(term0 + 0.5 * (term1 - term2))

        return preds

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
