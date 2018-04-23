'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class NN(object):
    def __init__(self, max_seq_len, vocab_size, word_embedding_dim, label_output_dim, label_prop, num_classify_hidden, args):
        self.max_seq_len = max_seq_len
        self.word_embedding_dim = word_embedding_dim
        self.label_output_dim = label_output_dim
        self.num_classify_hidden = num_classify_hidden
        self.label_prop = tf.constant(label_prop, dtype=tf.float32)
        self.batch_size = args.batch_size
        # self.dropout_keep_prob = args.dropout_keep_prob
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')
        #
        self.word_embedding = tf.get_variable('word_embedding', [vocab_size, word_embedding_dim], initializer=self.weight_initializer)
        #
        self.x_feature_id = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.x_feature_v = tf.placeholder(tf.float32, [None, self.max_seq_len])
        self.y = tf.placeholder(tf.float32, [None, self.label_output_dim])
        self.seqlen = tf.placeholder(tf.int32, [None])
        #self.label_embedding_id = tf.placeholder(tf.int32, [None])
        #self.label_prop = tf.placeholder(tf.float32, [None])

    def attention_layer(self, hidden_states, label_embeddings, hidden_dim, label_embedding_dim, seqlen, name_scope=None):
        # hidden_states: [batch_size, max_seq_len, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        with tf.variable_scope('att_layer'):
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
            # hidden_states: [batch_size, max_seq_len, hidden_dim]
            s_hidden = tf.multiply(s_mod, hidden_states)
            # s_hidden: [batch_size, max_seq_len, hidden_dim]
            # return z: [batch_size, hidden_dim]
            return tf.reduce_sum(s_hidden, axis=1)

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

    def build_model(self):
        # x: [batch_size, max_seq_len]
        # y: [batch_size, label_output_dim]
        word_embeddings_padding = tf.concat((tf.constant(0, dtype=tf.float32, shape=[1, self.word_embedding_dim]),
                                            self.word_embedding), axis=0)
        x = tf.nn.embedding_lookup(word_embeddings_padding, self.x_feature_id)
        # x: [batch_size, max_seq_len, word_embedding_dim]
        y = self.y
        # x_emb
        feature_v = tf.layers.batch_normalization(self.x_feature_v)
        x_emb = tf.reduce_sum(tf.multiply(x, tf.expand_dims(feature_v, -1)), axis=1)
        # ---------- attention --------------
        # with tf.name_scope('attention'):
        with tf.name_scope('output'):
            weight_1 = tf.get_variable('weight_1', [self.word_embedding_dim, self.num_classify_hidden],
                                       initializer =self.weight_initializer)
            bias_1 = tf.get_variable('bias_1', [self.num_classify_hidden], initializer=self.const_initializer)
            y_hidden = tf.nn.relu(tf.add(tf.matmul(x_emb, weight_1), bias_1))
            weight_2 = tf.get_variable('weight_2', [self.num_classify_hidden, self.label_output_dim],
                                       initializer=self.weight_initializer)
            #y_out = tf.nn.relu(tf.matmul(y_hidden, weight_2))
            y_out = tf.matmul(y_hidden, weight_2)
        # loss
        loss = tf.reduce_sum(
            tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_out), tf.expand_dims(self.label_prop, 0))
        )
        #if self.use_propensity:
        #    loss = tf.losses.sigmoid_cross_entropy(y, y_, weights=tf.expand_dims(self.label_prop, -1))
        #else:
        #    loss = tf.losses.sigmoid_cross_entropy(y, y_)
        return x_emb, tf.sigmoid(y_out), loss

