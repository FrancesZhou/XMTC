'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class XML_CNN(object):
    def __init__(self, max_seq_len, word_embedding_dim, filter_sizes, label_output_dim, hidden_dim, args):
        self.max_seq_len = max_seq_len
        self.word_embedding_dim = word_embedding_dim
        self.filter_sizes = filter_sizes
        self.label_output_dim = label_output_dim
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.hidden_dim = hidden_dim
        self.batch_size = args.batch_size
        self.dropout_keep_prob = args.dropout_keep_prob

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.word_embedding_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.label_output_dim])


    def build_model(self):
        # x: [batch_size, self.max_seq_len, self.embedding_dim]
        # y: [batch_size, self.label_output_dim]
        x = self.x
        x_expand = tf.expand_dims(x, axis=-1)
        y = self.y
        # dropout
        with tf.name_scope('dropout'):
            x_expand = tf.nn.dropout(x_expand, keep_prob=0.25)
        conv_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('convolution-pooling-{0}'.format(filter_size)):
                # ============= convolution ============
                filter = tf.get_variable('filter-{0}'.format(filter_size),
                                         [filter_size, self.word_embedding_dim, 1, self.num_filters],
                                         initializer=self.weight_initializer)
                conv = tf.nn.conv2d(x_expand, filter, strides=[1,1,1,1], padding='VALID', name='conv')
                b = tf.get_variable('b-{0}'.format(filter_size), [self.num_filters])
                conv_b = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')
                # conv_b: [batch_size, seqence_length-filter_size+1, 1, num_filters]
                # ============= dynamic max pooling =================
                pool_size = (self.max_seq_len - filter_size + 1) // self.pooling_units
                pool_out = tf.nn.max_pool(conv_b, ksize=[1, pool_size, 1, 1],
                                          strides=[1, pool_size, 1, 1], padding='VALID', name='dynamic-max-pooling')
                # pool_out: [batch_size, pooling_units, 1, num_filters]
                pool_out = tf.reshape(pool_out, [self.batch_size, -1])
                conv_outputs.append(pool_out)
        all_features = tf.concat(conv_outputs, -1)
        # dropout
        # with tf.name_scope('dropout'):
        #     fea_dropout = tf.nn.dropout(all_features, keep_prob=self.dropout_keep_prob)
        with tf.name_scope('output'):
            fea_dim = all_features.get_shape().as_list()[-1]
            # bottlenetck layer
            w_b = tf.get_variable('bottleneck_w', [fea_dim, self.hidden_dim], initializer=self.weight_initializer)
            l_hidden = tf.nn.relu(tf.matmul(all_features, w_b), 'relu')
            # dropout layer
            l_hidden_dropout = tf.nn.dropout(l_hidden, keep_prob=self.dropout_keep_prob)
            # output layer
            w_o = tf.get_variable('output_w', [self.hidden_dim, self.label_output_dim], initializer=self.weight_initializer)
            y_ = tf.nn.relu(tf.matmul(l_hidden_dropout, w_o), 'relu')

        # loss
        loss = tf.losses.sigmoid_cross_entropy(y, y_)
        return y_, y_, loss




