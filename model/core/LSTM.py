'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class LSTM(object):
    def __init__(self, max_seq_len, word_embedding_dim, hidden_dim, label_embedding_dim, num_classify_hidden, args):
        self.max_seq_len = max_seq_len
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.label_embedding_dim = label_embedding_dim
        self.num_classify_hidden = num_classify_hidden
        self.batch_size = args.batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.word_embedding_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, 2])
        self.seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.label_embeddings = tf.placeholder(tf.float32, [self.batch_size, self.label_embedding_dim])

    def attention_layer(self, hidden_states, label_embeddings, hidden_dim, label_embedding_dim, seqlen):
        # hidden_states: [batch_size, max_seq_len, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        # seqlen: [batch_size]
        with tf.variable_scope('att_layer'):
            w = tf.get_variable('w', [hidden_dim, label_embedding_dim], initializer=self.weight_initializer)
            # score: h*W*l
            # hidden_states: [batch_size, max_seq_len, hidden_dim]
            s = tf.reshape(tf.matmul(tf.reshape(hidden_states, [-1, hidden_dim]), w), [-1, self.max_seq_len, label_embedding_dim])
            # s: [batch_size, max_seq_len, label_embedding_dim]
            s = tf.matmul(s, tf.expand_dims(label_embeddings, axis=-1))
            # s: [batch_size, max_seq_len, 1]
            #
            mask = tf.where(tf.tile(tf.expand_dims(range(1, self.max_seq_len+1), axis=0), [self.batch_size, 1]) >
                            tf.tile(tf.expand_dims(seqlen, axis=-1), [1, self.max_seq_len]),
                            tf.ones([self.batch_size, self.max_seq_len])*self.neg_inf,
                            tf.zeros([self.batch_size, self.max_seq_len]))
            # mask: [batch_size, max_seq_len] with negative infinity in indices larger than seqlen
            s_mod = tf.nn.softmax(s + tf.expand_dims(mask, axis=-1), 1)
            # s_mod: [batch_size, max_seq_len, 1]
            # hidden_states: [batch_size, max_seq_len, hidden_dim]
            s_hidden = tf.multiply(s_mod, hidden_states)
            # s_hidden: [batch_size, max_seq_len, hidden_dim]
            # return z: [batch_size, hidden_dim]
            return tf.reduce_sum(s_hidden, axis=1)

    def classification(self, hidden_states, label_embeddings, hidden_dim, label_embedding_dim, seq_len):
        # hidden_states: [batch_size, max_seq_len, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        # seq_len: [batch_size]
        #
        # z: [batch_size, hidden_dim]
        # --------- attention -------------
        z = self.attention_layer(hidden_states, label_embeddings, hidden_dim, label_embedding_dim, seq_len)
        # --------- classification --------
        with tf.variable_scope('classify_layer'):
            with tf.variable_scope('z'):
                w = tf.get_variable('w', [hidden_dim, self.num_classify_hidden], initializer=self.weight_initializer)
                # z: [batch_size, hidden_dim]
                z_att = tf.matmul(z, w)
            with tf.variable_scope('label'):
                w = tf.get_variable('w', [label_embedding_dim, self.num_classify_hidden], initializer=self.weight_initializer)
                # label_embedding: [batch_size, label_embedding_dim]
                label_att = tf.matmul(label_embeddings, w)
            b = tf.get_variable('b', [self.num_classify_hidden], initializer=self.const_initializer)
            z_label_plus = tf.nn.relu(z_att + label_att) + b
            # z_label_plus: [batch_size, num_classify_hidden]
            #
            w_classify = tf.get_variable('w_classify', [self.num_classify_hidden, 2], initializer=self.weight_initializer)
            b_classify = tf.get_variable('b_classify', [2], initializer=self.const_initializer)
            wz_b_plus = tf.matmul(z_label_plus, w_classify) + b_classify
            # wz_b_plus: [batch_size, 2]
            return tf.nn.softmax(tf.nn.relu(wz_b_plus), -1)


    def build_model(self):
        # x: [batch_size, self.max_seq_len, self.word_embedding_dim]
        # y: [batch_size, 2]
        x = self.x
        y = self.y
        # ------------ LSTM ------------
        # activation: tanh(default)
        outputs, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.BasicLSTMCell(self.hidden_dim), x, dtype=tf.float32,
                                       sequence_length=self.seqlen)
        outputs = tf.stack(outputs)
        print('outputs_shape : ', outputs.get_shape().as_list())
        # outputs: [batch_size, max_seq_len, hidden_dim]
        # ----------- get x-embedding ----------
        # Indexing
        index = tf.range(0, self.batch_size)*self.max_seq_len + (self.seqlen - 1)
        x_emb = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
        print('after indexing, shape of x_emb : ', x_emb.get_shape().as_list())
        # ------------ attention and classification --------------
        # outputs: [batch_size, max_seq_len, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        y_ = self.classification(outputs, self.label_embeddings, self.hidden_dim, self.label_embedding_dim, self.seqlen)
        y_pre = y_
        y_tar = y
        loss = tf.losses.sigmoid_cross_entropy(y_tar, y_pre)
        return x_emb, y_pre[:, 1], loss




