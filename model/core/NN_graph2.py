'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import categorical_crossentropy

class NN_graph2(object):
    def __init__(self, max_seq_len, vocab_size, word_embedding_dim, label_embedding, num_classify_hidden,
                 args):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.num_classify_hidden = num_classify_hidden
        self.label_num = label_embedding.shape[0]
        self.label_embedding_dim = label_embedding.shape[-1]
        self.use_attention = args.use_attention
        self.neg_samp = args.neg_samp
        self.batch_size = args.batch_size
        self.active_feature_num = args.active_feature_num
        # self.dropout_keep_prob = args.dropout_keep_prob
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')
        #
        self.word_embedding = tf.get_variable('word_embedding', [vocab_size, word_embedding_dim], initializer=self.weight_initializer)
        if args.random_label_embedding:
            self.label_embedding = tf.get_variable('label_embedding', [self.label_num, self.label_embedding_dim], initializer=self.weight_initializer)
        elif args.fix_label_embedding:
            self.label_embedding = tf.constant(label_embedding, dtype=tf.float32)
        else:
            self.label_embedding = tf.get_variable('label_embedding', initializer=tf.constant(label_embedding, dtype=tf.float32))
        #
        self.use_graph = args.use_graph
        #
        self.x_feature_id = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.x_feature_v = tf.placeholder(tf.float32, [None, self.max_seq_len])
        self.y = tf.placeholder(tf.float32, [None])
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.label_embedding_id = tf.placeholder(tf.int32, [None])
        self.label_prop = tf.placeholder(tf.float32, [None])
        #
        self.label_active_feature = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32, shape=[32, self.active_feature_num]),
                                                                [None, self.active_feature_num])
        #
        self.gl1 = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32, shape=[3]), [None])
        self.gl2 = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32, shape=[3]), [None])
        self.gy = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[3]), [None])

    def attention_layer(self, hidden_states, label_embeddings, hidden_dim, label_embedding_dim, seqlen, name_scope=None):
        # hidden_states: [batch_size, max_seq_len, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        with tf.variable_scope('att_layer', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [hidden_dim, label_embedding_dim], initializer=self.weight_initializer)
            # hidden_states: [batch_size, max_seq_len, hidden_dim]
            # label_embeddings: [batch_size, label_embedding_dim]
            # score: h*W*l
            s = tf.matmul(tf.reshape(tf.matmul(tf.reshape(hidden_states, [-1, hidden_dim]), w), [-1, self.max_seq_len, label_embedding_dim]),
                          tf.expand_dims(label_embeddings, axis=-1))
            # s: [batch_size, max_seq_len, 1]
            #
            mask = tf.where(tf.tile(tf.expand_dims(range(1, self.max_seq_len + 1), axis=0), [self.batch_size, 1]) >
                            tf.tile(tf.expand_dims(seqlen, axis=-1), [1, self.max_seq_len]),
                            tf.ones([self.batch_size, self.max_seq_len]) * self.neg_inf,
                            tf.zeros([self.batch_size, self.max_seq_len]))
            # mask: [batch_size, max_seq_len] with negative infinity in indices larger than seqlen
            s_mod = tf.nn.softmax(s + tf.expand_dims(mask, axis=-1), 1)
            # s_mod: [batch_size, max_seq_len, 1]
            return tf.squeeze(s_mod)
            # s_hidden = tf.multiply(s_mod, hidden_states)
            # return tf.reduce_sum(s_hidden, axis=1)
            # hidden_states: [batch_size, max_seq_len, hidden_dim]
            # s_hidden: [batch_size, max_seq_len, hidden_dim]
            # return z: [batch_size, hidden_dim]


    def classification_layer(self, features, label_embeddings, hidden_dim, label_embedding_dim):
        # features: [batch_size, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        #
        with tf.variable_scope('classification_layer'):
            with tf.variable_scope('features'):
                w_fea = tf.get_variable('w_fea', [hidden_dim, self.num_classify_hidden],
                                        initializer=self.weight_initializer)
                # features: [batch_size, hidden_dim]
                fea_att = tf.matmul(features, w_fea)
            with tf.variable_scope('label'):
                w_label = tf.get_variable('w_label', [label_embedding_dim, self.num_classify_hidden],
                                          initializer=self.weight_initializer)
                # label_embedding: [batch_size, label_embedding_dim]
                label_att = tf.matmul(label_embeddings, w_label)
            b = tf.get_variable('b', [self.num_classify_hidden], initializer=self.const_initializer)
            fea_label_plus = tf.add(fea_att, label_att)
            fea_label_plus_b = tf.nn.relu(tf.add(fea_label_plus, b))
            # fea_label_plus_b: [batch_size, num_classify_hidden]
            #
            with tf.variable_scope('classify'):
                w_classify = tf.get_variable('w_classify', [self.num_classify_hidden, 1],
                                             initializer=self.weight_initializer)
                # b_classify = tf.get_variable('b_classify', [1], initializer=self.const_initializer)
                out = tf.matmul(fea_label_plus_b, w_classify)
                # out = tf.add(wz_b_plus, b_classify)
                # out: [batch_size, 1]
        return tf.squeeze(out)

    def gaussian_noise_layer(input_layer, std=0.2):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return tf.add(input_layer, noise)

    def pre_build_model(self):
        # x_feature_id: [batch_size, max_seq_len]
        # x_feature_v: [batch_size, max_seq_len]
        # y: [batch_size]
        #
        # label embeddings
        label_embeddings = tf.nn.embedding_lookup(self.label_embedding, self.label_embedding_id)
        # y
        y = self.y
        #y = tf.multiply(self.y, self.label_prop)
        # ---------- x -------------
        word_embeddings_padding = tf.concat((tf.constant(0, dtype=tf.float32, shape=[1, self.word_embedding_dim]),
                                     self.word_embedding), axis=0)
        x_feature_id = self.x_feature_id
        x = tf.nn.embedding_lookup(word_embeddings_padding, x_feature_id)
        # x: [batch_size, max_seq_len, word_embedding_dim]
        # normalize feature_v
        #feature_v = tf.divide(self.x_feature_v, tf.norm(self.x_feature_v, 2, axis=-1, keepdims=True))
        #feature_v = tf.divide(self.x_feature_v, tf.reduce_sum(self.x_feature_v, -1, keepdims=True))
        feature_v = self.x_feature_v
        feature_v = tf.layers.batch_normalization(feature_v)
        #feature_v = tf.contrib.layers.dropout(feature_v, keep_prob=0.5)
        if self.use_attention:
            with tf.name_scope('attention'):
                att_weight = self.attention_layer(x, label_embeddings,
                                                 self.word_embedding_dim, self.label_embedding_dim,
                                                 self.seqlen)
                # att_weight: [batch_size, max_seq_len)
                feature_v = tf.multiply(feature_v, att_weight)
        # ---------- feature embeddings -----------
        x_emb = tf.reduce_sum(tf.multiply(x, tf.expand_dims(feature_v, -1)), axis=1)
        # x_emb: [batch_size, word_embedding_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        x_label_concat = tf.concat([x_emb, label_embeddings], axis=-1)
        # ---------- output layer ----------
        #y_hidden = tf.layers.dense(x_label_concat, self.num_classify_hidden, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.contrib.layers.l2_regularizer, name='pre_dense_0')
        weight_1 = tf.get_variable('weight_hidden_1', [self.word_embedding_dim + self.label_embedding_dim, self.num_classify_hidden], initializer=self.weight_initializer)
        y_hidden = tf.nn.relu(tf.matmul(x_label_concat, weight_1))
        #y_hidden = tf.contrib.layers.dropout(y_hidden, keep_prob=0.5)
        y_hidden = tf.layers.batch_normalization(y_hidden)
        #y_hidden = tf.contrib.layers.dropout(y_hidden, keep_prob=0.5)
        #y_out = tf.layers.dense(y_hidden, 1, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer, name='pre_dense_1')
        weight_2 = tf.get_variable('weight_hidden_2', [self.num_classify_hidden, 2], initializer=self.weight_initializer)
        y_out = tf.matmul(y_hidden, weight_2)
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.cast(y, dtype=tf.int32), 2), logits=y_out, weights=self.label_prop) + tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2)
        #loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.squeeze(y_out)), self.label_prop)) + tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2)
        #loss = tf.nn.l2_loss(y - y_out, name='l2_loss')
        # ---------- graph context loss ---------------
        if self.use_graph:
            with tf.variable_scope('graph_embedding', reuse=tf.AUTO_REUSE):
                gl1 = tf.nn.embedding_lookup(self.label_embedding, self.gl1)
                if self.neg_samp:
                    gl2 = tf.nn.embedding_lookup(
                        tf.get_variable('context_embedding', [self.label_num, self.label_embedding_dim],
                                        initializer=self.weight_initializer),
                        self.gl2)
                    l_gy = tf.multiply(gl1, gl2)
                    g_loss = tf.reduce_mean(-tf.log(tf.sigmoid(tf.multiply(tf.reduce_sum(l_gy, axis=1), self.gy))))
                else:
                    l_gy = tf.layers.dense(gl1, self.label_embedding_dim, activation=tf.nn.softmax, use_bias=False)
                    g_loss = tf.reduce_mean(categorical_crossentropy(tf.one_hot(self.gl2, self.label_embedding_dim), l_gy))
        else:
            g_loss = 0
        # ---------- get feature_gradient -------------
        word_grads = tf.gradients(loss, [self.word_embedding])[0]
        word_abs_grads = tf.abs(word_grads)
        sum_word_grads = tf.reduce_sum(word_abs_grads, axis=-1)
        #print 'shape of sum_word_grads'
        #print sum_word_grads.get_shape().as_list()
        return x_emb, y_out, sum_word_grads, loss, g_loss

    def build_model(self):
        # x_feature_id: [batch_size, max_seq_len]
        # x_feature_v: [batch_size, max_seq_len]
        # y: [batch_size]
        #
        # label embeddings
        label_embeddings = tf.nn.embedding_lookup(self.label_embedding, self.label_embedding_id)
        # y
        y = self.y
        #y = tf.multiply(self.y, self.label_prop)
        # ---------- x -------------
        word_embeddings_padding = tf.concat((tf.constant(0, dtype=tf.float32, shape=[1, self.word_embedding_dim]),
                                     self.word_embedding), axis=0)
        x = tf.nn.embedding_lookup(word_embeddings_padding, self.x_feature_id)
        # x: [batch_size, max_seq_len, word_embedding_dim]
        #feature_v = self.gaussian_noise_layer(tf.cast(feature_v, dtype=tf.float32))
        feature_v = self.x_feature_v
        feature_v = tf.layers.batch_normalization(feature_v)
        if self.use_attention:
            with tf.name_scope('attention'):
                att_weight = self.attention_layer(x, label_embeddings,
                                                 self.word_embedding_dim, self.label_embedding_dim,
                                                 self.seqlen)
                # att_weight: [batch_size, max_seq_len)
                feature_v = tf.multiply(feature_v, att_weight)
        # ---------- feature embeddings -----------
        x_emb = tf.reduce_sum(tf.multiply(x, tf.expand_dims(feature_v, -1)), axis=1)
        # x_emb: [batch_size, word_embedding_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        #
        feature_label_embeddings = tf.reduce_sum(tf.nn.embedding_lookup(word_embeddings_padding, self.label_active_feature), axis=1)
        x_label_concat = tf.concat([x_emb, label_embeddings], axis=-1)
        x_label_concat = tf.concat([x_label_concat, feature_label_embeddings], axis=-1)
        # ---------- output layer ----------
        #y_hidden = tf.layers.dense(x_label_concat, self.num_classify_hidden, activation=tf.nn.relu, use_bias=True, kernel_regularizer=tf.contrib.layers.l2_regularizer, name='dense_0')
        weight_1 = tf.get_variable('train_weight_1', [self.word_embedding_dim*2 + self.label_embedding_dim, self.num_classify_hidden], initializer=self.weight_initializer)
        y_hidden = tf.nn.relu(tf.matmul(x_label_concat, weight_1))
        #y_hidden = tf.contrib.layers.dropout(y_hidden, keep_prob=0.5)
        y_hidden = tf.layers.batch_normalization(y_hidden)
        weight_2 = tf.get_variable('train_weight_2', [self.num_classify_hidden, 2], initializer=self.weight_initializer)
        y_out = tf.matmul(y_hidden, weight_2)
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.cast(y, dtype=tf.int32), 2), logits=y_out, weights=self.label_prop) + tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2)
        #y_out = tf.layers.dense(y_hidden, 1, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer, name='dense_1')
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.squeeze(y_out)) + tf.reduce_sum(tf.losses.get_regularization_losses())
        #loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.squeeze(y_out)), self.label_prop)) + tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2)
        #loss = tf.nn.l2_loss(y - y_out, name='l2_loss')
        # ---------- graph context loss ---------------
        if self.use_graph:
            gl1 = tf.nn.embedding_lookup(self.label_embedding, self.gl1)
            if self.neg_samp:
                gl2 = tf.nn.embedding_lookup(
                    tf.get_variable('context_embedding', [self.label_num, self.label_embedding_dim],
                                    initializer=self.weight_initializer),
                    self.gl2)
                l_gy = tf.multiply(gl1, gl2)
                g_loss = tf.reduce_mean(-tf.log(tf.sigmoid(tf.multiply(tf.reduce_sum(l_gy, axis=1), self.gy))))
            else:
                l_gy = tf.layers.dense(gl1, self.label_embedding_dim, activation=tf.nn.softmax, use_bias=False)
                g_loss = tf.reduce_mean(categorical_crossentropy(tf.one_hot(self.gl2, self.label_embedding_dim), l_gy))
        else:
            g_loss = 0
        # ---------- get feature_gradient -------------
        # word_grads = tf.gradients(loss, [self.word_embedding])[0]
        # sum_word_grads = tf.sparse_reduce_sum(word_grads, axis=-1)
        # print 'shape of sum_word_grads'
        # print sum_word_grads.get_shape().as_list()
        return x_emb, y_out, loss, g_loss


