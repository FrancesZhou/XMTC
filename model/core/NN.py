'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class NN(object):
    def __init__(self, max_seq_len, vocab_size, word_embedding_dim, label_embedding, num_classify_hidden, args):
        self.max_seq_len = max_seq_len
        self.word_embedding_dim = word_embedding_dim
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.num_classify_hidden = num_classify_hidden
        self.label_embedding_dim = label_embedding.shape[-1]
        self.batch_size = args.batch_size
        # self.dropout_keep_prob = args.dropout_keep_prob
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')
        #
        self.word_embedding = tf.get_variable('word_embedding', [vocab_size, word_embedding_dim])
        self.label_embedding = tf.constant(label_embedding, dtype=tf.float32)
        #
        self.x = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.y = tf.placeholder(tf.float32, [None])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.label_embedding_id = tf.placeholder(tf.int32, [None])
        self.label_prop = tf.placeholder(tf.float32, [None])

    def attention_layer(self, hidden_states, label_embeddings, hidden_dim, label_embedding_dim, seqlen, name_scope=None):
        # hidden_states: [batch_size, max_seq_len, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        with tf.variable_scope(name_scope + 'att_layer'):
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
        # x: [batch_size, self.max_seq_len]
        # y: [batch_size]
        x = tf.nn.embedding_lookup(self.word_embedding, self.x)
        # x: [batch_size, self.max_seq_len, word_embedding_dim]
        label_embeddings = tf.nn.embedding_lookup(self.label_embedding, self.label_embedding_id)
        y = self.y
        # x_emb
        x_emb = tf.reduce_max(x, axis=1)
        # ---------- attention --------------
        with tf.name_scope('attention'):
            x_lbl_fea = self.attention_layer(x, label_embeddings,
                                             self.word_embedding_dim, self.label_embedding_dim,
                                             self.seq_len)
        with tf.name_scope('output'):
            fea_dim = x_lbl_fea.get_shape().as_list()[-1]
            y_ = self.classification_layer(x_lbl_fea, label_embeddings, fea_dim, self.label_embedding_dim)
        # loss
        loss = tf.reduce_sum(
            tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_), self.label_prop)
        )
        # if self.use_propensity:
        #     loss = tf.losses.sigmoid_cross_entropy(y, y_, weights=tf.expand_dims(self.label_prop, -1))
        # else:
        #     loss = tf.losses.sigmoid_cross_entropy(y, y_)
        return x_emb, tf.sigmoid(y_), loss

