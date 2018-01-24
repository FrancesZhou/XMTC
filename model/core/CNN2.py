'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class CNN2(object):
    def __init__(self, max_seq_len, output_dim, word_embedding, filter_sizes, label_embedding, hidden_dim, args):
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim
        #self.topk = topk
        self.word_embedding_dim = word_embedding.shape[-1]
        self.filter_sizes = filter_sizes
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.hidden_dim = hidden_dim
        self.label_embedding_dim = label_embedding.shape[-1]
        #self.batch_size = batch_size
        self.dropout_keep_prob = args.dropout_keep_prob
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        #
        self.word_embedding = tf.constant(word_embedding, dtype=tf.float32)
        self.pretrained_label_embedding = tf.constant(label_embedding, dtype=tf.float32)
        #
        self.x = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.y = tf.placeholder(tf.float32, [None, output_dim])
        self.label_embedding_id = tf.placeholder(tf.int32, [None, output_dim])

    def attention_layer(self, hidden_states, label_embeddings, hidden_dim, label_embedding_dim, name_scope=None):
        # label_embeddings: [batch_size, output_dim, label_embedding_dim]
        # hidden_states: [batch_size, num_hiddens, hidden_dim]
        with tf.variable_scope(name_scope + 'att_layer'):
            w = tf.get_variable('w', [label_embedding_dim, hidden_dim], initializer=self.weight_initializer)
            # hidden_states: [batch_size, num, hidden_dim]
            # score: l*W*h
            s = tf.matmul(tf.reshape(tf.matmul(tf.reshape(label_embeddings, [-1, label_embedding_dim]), w), [-1, self.output_dim, hidden_dim]),
                          tf.transpose(hidden_states, perm=[0, 2, 1]))
            # s: [batch_size, output_dim, num_hiddens]
            s = tf.nn.softmax(s, -1)
            # s: [batch_size, output_dim, num_hiddens]
            # hidden_states: [batch_size, num_hiddens, hidden_dim]
            # s_hidden: [batch_size, output_dim, num_hiddens, hidden_dim]
            s_hidden = tf.multiply(tf.expand_dims(s, axis=-1), tf.expand_dims(hidden_states, axis=1))
            # return [batch_size, output_dim, hidden_dim]
            return tf.reduce_sum(s_hidden, axis=2)

    def competitive_layer(self, all_num_filters, all_features, topk, factor=6.26):
        # all_features: [batch_size, output_dim, all_num_filters]
        x = tf.transpose(all_features, perm=[0, 2, 1])
        # x: [batch_size, all_num_filters, output_dim]
        x = tf.reshape(x, [-1, self.output_dim])
        # x: [batch_size*all_num_filters, output]
        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2
        # P
        values, indices = tf.nn.top_k(P, topk/2)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk/2])
        full_indices = tf.stack([my_range_repeated, indices], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
        # N
        values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0.,
                                     validate_indices=False)
        # tmp
        P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True)
        N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]),
                                     default_value=0., validate_indices=False)
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]),
                                     default_value=0., validate_indices=False)
        res = P_reset - N_reset
        # [batch_size*all_num_filters, output_dim] ->
        # [batch_size, all_num_filters, output_dim] ->
        # return [batch_size, output_dim, all_num_filters]
        return tf.transpose(tf.reshape(res, [-1, all_num_filters, self.output_dim]), perm=[0, 2, 1])
        #P_tmp = tf.reshape(factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True), [-1, all_num_filters, 1])
        #N_tmp = tf.reshape(factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True), [-1, all_num_filters, 1])
        #P_reset = tf.reshape(P_reset, [-1, all_num_filters, self.output_dim])
        #N_reset = tf.reshape(N_reset, [-1, all_num_filters, self.output_dim])

    def output_layer(self, x_emb, comp_all_features, all_num_filters):
        # x_emb : [batch_size, all_num_filters]
        # comp_all_features : [batch_size, output_dim, all_num_filters]
        #x_emb_tile = tf.tile(tf.expand_dims(x_emb, 1), [1, self.output_dim, 1])
        # x_emb_tile: [batch_size, output_dim, all_num_filters]
        #res_all_features = tf.nn.relu(tf.subtract(x_emb_tile, comp_all_features))
        with tf.variable_scope('output_layer'):
            w_output = tf.get_variable('w_output', [self.output_dim, all_num_filters], initializer=self.weight_initializer)
            b_output = tf.get_variable('b_output', [self.output_dim], initializer=self.const_initializer)
            wf = tf.reduce_sum(tf.multiply(comp_all_features, w_output), -1)
            #wf = tf.reduce_sum(tf.multiply(res_all_features, w_output), -1)
            wf_b_plus = tf.add(wf, b_output)
        # wf_b_plus: [batch_size, output_dim]
        return wf_b_plus

    def score(self, x_emb, all_num_filters, label_embeddings, label_embedding_dim, hidden_dim):
        # x_emb: [batch_size, all_num_filters]
        # label_embeddings: [batch_size, output_dim, label_embedding_dim]
        with tf.variable_scope('socre'):
            with tf.variable_scope('x-embedding'):
                w_x = tf.get_variable('w_x', [all_num_filters, hidden_dim], initializer=self.weight_initializer)
                # x_emb: [batch_size, all_num_filters]
                x_project = tf.matmul(x_emb, w_x)
                # x_project : [batch_size, hidden_dim]
            with tf.variable_scope('label'):
                w_label = tf.get_variable('w_label', [label_embedding_dim, hidden_dim], initializer=self.weight_initializer)
                # label_embedding: [batch_size, output_dim, label_embedding_dim]
                label_project = tf.reshape(tf.matmul(tf.reshape(label_embeddings, [-1, label_embedding_dim]), w_label), [-1, self.output_dim, hidden_dim])
                # label_project : [batch_size, output_dim, hidden_dim]
            b = tf.get_variable('b', [hidden_dim], initializer=self.const_initializer)
            x_label_plus = tf.add(tf.tile(tf.expand_dims(x_project, 1), [1, self.output_dim, 1]),
                                  label_project)
            x_label_plus_b = x_label_plus + b
            # x_label_plus_b: [batch_size, output_dim, hidden_dim]
            # output socre
            with tf.variable_scope('output'):
                w_output = tf.get_variable('w_output', [hidden_dim, 1], initializer=self.weight_initializer)
                #b_output = tf.get_variable('b_output', [self.output_dim], initializer=self.const_initializer)
                wx_output = tf.reshape(tf.matmul(tf.reshape(x_label_plus_b, [-1, hidden_dim]), w_output), [-1, self.output_dim])
                #score = wx_output + b_output
                score = wx_output
                #score = tf.reshape(wx_b_plus, [-1, self.output_dim])
                # score: [batch_size, output_dim]
        return score

    def build_model(self):
        # x: [batch_size, self.max_seq_len]
        # y: [batch_size, output_dim]
        # label: [batch_size, output_dim]
        # x = self.x
        x = tf.nn.embedding_lookup(self.word_embedding, self.x)
        # x: [batch_size, self.max_seq_len, word_embedding_dim]
        self.label_embedding = tf.get_variable('label_embedding', initializer=self.pretrained_label_embedding)
        label_embeddings = tf.nn.embedding_lookup(self.label_embedding, self.label_embedding_id)
        # label_embeddings: [batch_size, output_dim, label_embedding_dim]
        x_expand = tf.expand_dims(x, axis=-1)
        # x: [batch_size, self.max_seq_len, word_embedding_dim, 1]
        y = self.y
        # dropout
        # x_expand = tf.nn.dropout(x_expand, keep_prob=0.25)
        max_pool_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('convolution-pooling-{0}'.format(filter_size)) as name_scope:
                # ============= convolution ============
                filter = tf.get_variable('filter-{0}'.format(filter_size),
                                         [filter_size, self.word_embedding_dim, 1, self.num_filters],
                                         initializer=self.weight_initializer)
                conv = tf.nn.conv2d(x_expand, filter, strides=[1,1,1,1], padding='VALID', name='conv')
                b = tf.get_variable('b-{0}'.format(filter_size), [self.num_filters])
                conv_b = tf.nn.bias_add(conv, b)
                # conv_b: [batch_size, seqence_length-filter_size+1, 1, num_filters]
                # ============= max pooling for x-embedding =========
                pool_emb = tf.nn.max_pool(conv_b, ksize=[1, self.max_seq_len-filter_size+1, 1, 1],
                                          strides=[1, 1, 1, 1], padding='VALID', name='max-pooling')
                # pool_emb: [batch_size, 1, 1, num_filters]
                max_pool_outputs.append(tf.squeeze(pool_emb, [1, 2]))
                # ============== dynamic max pooling =================
                '''
                pool_size = (self.max_seq_len - filter_size + 1) // self.pooling_units
                pool_out = tf.nn.max_pool(conv_b, ksize=[1, pool_size, 1, 1],
                                          strides=[1, pool_size, 1, 1], padding='VALID', name='dynamic-max-pooling')
                # pool_out: [batch_size, pooling_units, 1, num_filters]
                dynamic_pool_outputs.append(tf.squeeze(pool_out, [-2]))
                '''
                # ============= attention ===============
                '''
                pool_squeeze = tf.squeeze(pool_out, [-2])
                print [None, self.pooling_units, self.num_filters]
                print pool_squeeze.get_shape().as_list()
                num_hiddens = (self.max_seq_len - filter_size + 1) // pool_size
                # pool_squeeze: [batch_size, num_hiddens, num_filters]
                print num_hiddens
                #
                l_feature = self.attention_layer(pool_squeeze, label_embeddings, self.num_filters, self.label_embedding_dim, name_scope=name_scope)
                # [batch_size, output_dim, hidden_dim]
                # l_feature: [batch_size, output_dim, num_filters]
                conv_atten_outputs.append(l_feature)
                '''
        x_emb = tf.concat(max_pool_outputs, -1)
        all_num_filters = self.num_filters * len(self.filter_sizes)
        score = self.score(x_emb, all_num_filters, label_embeddings, self.label_embedding_dim, self.hidden_dim)
        # all_features = tf.concat(conv_atten_outputs, -1)
        # all_features: [batch_size, output_dim, all_num_filters]
        # ------------- dropout ------------
        # with tf.name_scope('dropout'):
        #     all_features = tf.nn.dropout(all_features, keep_prob=self.dropout_keep_prob)
        # ------------- competitive ----------------
        # comp_all_features = self.competitive_layer(all_num_filters, all_features, self.topk)
        # comp_all_features : [batch_size, output_dim, all_num_filters]
        # output
        #y_ = self.output_layer(x_emb, comp_all_features, all_num_filters)
        # loss
        # loss = tf.losses.sigmoid_cross_entropy(y, y_)
        # y: [batch_size, output_dim]
        # score: [batch_size, output_dim]
        # ------------ rank loss -------------
        # y_pair = tf.tile(tf.expand_dims(y, -1), [1, 1, self.output_dim]) - tf.tile(tf.expand_dims(y, 1), [1, self.output_dim, 1])
        # score_pair = tf.tile(tf.expand_dims(score, -1), [1, 1, self.output_dim]) - tf.tile(tf.expand_dims(score, 1), [1, self.output_dim, 1])
        # pair: [batch_size, output_dim, output_dim]
        # mask = tf.subtract(tf.ones([self.output_dim, self.output_dim]), tf.diag(tf.ones([self.output_dim])))
        # mask: [output_dim, output_dim]
        loss_reg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=y))
        # loss_rank = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score_pair, labels=y_pair) * mask)
        pos_indices = tf.where(tf.greater(y, tf.constant(0, dtype=tf.float32)))
        neg_indices = tf.where(tf.equal(y, tf.constant(0, dtype=tf.float32)))
        sigmoid_score = tf.sigmoid(score)
        pos_score = tf.sparse_to_dense(pos_indices, tf.shape(y, out_type=tf.int64), tf.gather_nd(sigmoid_score, pos_indices), default_value=0.,
                                       validate_indices=False)
        neg_score = tf.sparse_to_dense(neg_indices, tf.shape(y, out_type=tf.int64), tf.gather_nd(sigmoid_score, neg_indices), default_value=0.,
                                       validate_indices=False)
        loss_sep_rank = tf.maximum(tf.constant(0, dtype=tf.float32), 1 - tf.reduce_min(pos_score, -1) + tf.reduce_max(neg_score, -1))
        #loss = loss_rank + loss_reg
        loss = tf.reduce_mean(loss_sep_rank) + loss_reg
        return x_emb, score, loss




