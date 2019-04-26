#!/usr/local/bin/bash
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DEPFM(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'depfm'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_features = data_config['n_features']

        # the settings of feature graph laplacian.
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.n_fold = 100

        # the basic settings.
        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # the settings of the deep component.
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

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

        # dropout: node dropout (adopted on the ego-networks); message dropout (adopted on the convolution operations).
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        # Initialize the model parameters.
        self.weights = self._init_weights()

        if self.alg_type in ['org']:
            self.var_a_factor = self._create_mean_pooling_graph_embed()
        elif self.alg_type in ['bi']:
            self.var_a_factor = self._create_bi_interaction_graph_embed()
        elif self.alg_type in ['gcn']:
            self.var_a_factor = self._create_gcn_interaction_graph_embed()
        elif self.alg_type in ['ego']:
            self.var_a_factor = self._create_ego_interaction_graph_embed()

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

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def _create_mean_pooling_graph_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = self.weights['var_factor']
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        return all_embeddings

    def _create_bi_interaction_graph_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = self.weights['var_factor']
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                sum_emb = tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings)
                sum_emb = tf.square(sum_emb)

                square_emb = tf.sparse_tensor_dense_matmul(tf.square(A_fold_hat[f]), tf.square(embeddings))
                temp_embed.append(0.5*(sum_emb-square_emb))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        return all_embeddings

    def _create_gcn_interaction_graph_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        pre_embeddings = self.weights['var_factor']

        for k in range(0, self.n_layers):
            # line 1 in algorithm 1 [RM-GCN, KDD'2018], transforming the embeddings first
            embeddings = tf.nn.relu(tf.matmul(pre_embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])

            # line 1 in algorithm 1 [RM-GCN, KDD'2018], aggregator layer: weighted sum
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)

            # line 2 in algorithm 1 [RM-GCN, KDD'2018], aggregating the previsou embeddings
            embeddings = tf.concat([pre_embeddings, embeddings], 1)
            pre_embeddings = tf.nn.relu(tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k])

        return pre_embeddings

    def _create_ego_interaction_graph_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = self.weights['var_factor']
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            ego_embeddings = tf.nn.leaky_relu(
                tf.matmul(ego_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [ego_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        return all_embeddings



    def _create_log_loss(self, sp_feats):
        self.preds = self._create_bi_predictions(sp_feats)

        log_loss = tf.reduce_mean(tf.losses.log_loss(labels=tf.reshape(self.sp_labels, [-1]),
                                      predictions=tf.reshape(self.preds, [-1])))

        reg_loss = self.regs[0] * tf.nn.l2_loss(self.weights['var_linear']) + \
                      self.regs[1] * tf.nn.l2_loss(self.var_a_factor)

        # reg_loss = self.regs[0] * tf.nn.l2_loss(self.weights['var_linear']) + \
        #            self.regs[1] * tf.nn.l2_loss(self.weights['var_factor'])
        #
        # for k in range(self.n_layers):
        #     reg_loss += self.regs[-1] * (tf.nn.l2_loss(self.weights['W_gc_%d' % k]) +
        #                                  tf.nn.l2_loss(self.weights['b_gc_%d' % k]))

        return log_loss, reg_loss

    def _create_bi_predictions(self, feats):
        # Linear terms.
        term0 = tf.sparse_tensor_dense_matmul(feats, self.weights['var_linear'])
        # Interaction terms w.r.t. first sum then square.
        #   e.g., sum_{k from 1 to K}{(v1k+v2k)**2}
        sum_emb = tf.sparse_tensor_dense_matmul(feats, self.var_a_factor)
        term1 = tf.square(tf.reduce_sum(sum_emb, axis=1, keepdims=True))

        # Interaction terms w.r.t. first square then sum.
        #   e.g., sum_{k from 1 to K}{v1k**2 + v2k**2}
        square_emb = tf.sparse_tensor_dense_matmul(tf.square(feats), tf.square(self.var_a_factor))
        term2 = tf.reduce_sum(square_emb, axis=1, keepdims=True)

        preds = tf.sigmoid(term0 + 0.5 * (term1 - term2))

        return preds

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_features) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_features
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
            """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

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