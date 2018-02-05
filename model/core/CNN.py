'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class CNN(object):
    def __init__(self, max_seq_len, word_embedding, filter_sizes, label_embedding, num_classify_hidden, args):
        self.max_seq_len = max_seq_len
        self.word_embedding_dim = word_embedding.shape[-1]
        self.filter_sizes = filter_sizes
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.num_classify_hidden = num_classify_hidden
        self.label_embedding_dim = label_embedding.shape[-1]
        self.dropout_keep_prob = args.dropout_keep_prob
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        #
        self.word_embedding = tf.constant(word_embedding, dtype=tf.float32)
        #self.label_embedding = tf.constant(label_embedding, dtype=tf.float32)
        self.label_embedding = tf.get_variable('label_embedding', initializer=label_embedding)
        #
        # self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.word_embedding_dim])
        # self.seqlen = tf.placeholder(tf.int32, [self.batch_size])
        # self.label_prop = tf.placeholder(tf.float32, [self.batch_size])
        #
        # self.x = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len])
        # self.y = tf.placeholder(tf.float32, [self.batch_size, 2])
        # self.label_embedding_id = tf.placeholder(tf.int32, [self.batch_size])
        self.x = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.y = tf.placeholder(tf.float32, [None])
        self.label_embedding_id = tf.placeholder(tf.int32, [None])
        self.label_prop = tf.placeholder(tf.float32, [None])

    def attention_layer(self, hidden_states, label_embeddings, num_hiddens, hidden_dim, label_embedding_dim, name_scope=None):
        # hidden_states: [batch_size, num_hiddens, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        with tf.variable_scope(name_scope + 'att_layer'):
            w = tf.get_variable('w', [hidden_dim, label_embedding_dim], initializer=self.weight_initializer)
            # hidden_states: [batch_size, num, hidden_dim]
            # label_embeddings: [batch_size, label_embedding_dim]
            # score: h*W*l
            s = tf.matmul(tf.reshape(tf.matmul(tf.reshape(hidden_states, [-1, hidden_dim]), w), [-1, num_hiddens, label_embedding_dim]),
                          tf.expand_dims(label_embeddings, axis=-1))
            # s: [batch_size, num, 1]
            s = tf.nn.softmax(s, 1)
            # s: [batch_size, num, 1]
            # hidden_states: [batch_size, num, hidden_dim]
            # s_hidden: [batch_size, num, hidden_dim]
            s_hidden = tf.multiply(s, hidden_states)
            return tf.reduce_sum(s_hidden, axis=1)

    def classification_layer(self, features, label_embeddings, hidden_dim, label_embedding_dim):
        # features: [batch_size, hidden_dim]
        # label_embeddings: [batch_size, label_embedding_dim]
        #
        with tf.variable_scope('classification_layer'):
            with tf.variable_scope('features'):
                w_fea = tf.get_variable('w_fea', [hidden_dim, self.num_classify_hidden], initializer=self.weight_initializer)
                # features: [batch_size, hidden_dim]
                fea_att = tf.matmul(features, w_fea)
            with tf.variable_scope('label'):
                w_label = tf.get_variable('w_label', [label_embedding_dim, self.num_classify_hidden], initializer=self.weight_initializer)
                # label_embedding: [batch_size, label_embedding_dim]
                label_att = tf.matmul(label_embeddings, w_label)
            b = tf.get_variable('b', [self.num_classify_hidden], initializer=self.const_initializer)
            fea_label_plus = tf.add(fea_att, label_att)
            fea_label_plus_b = tf.nn.relu(tf.add(fea_label_plus, b))
            # fea_label_plus_b: [batch_size, num_classify_hidden]
            #
            with tf.variable_scope('classify'):
                w_classify = tf.get_variable('w_classify', [self.num_classify_hidden, 1], initializer=self.weight_initializer)
                #b_classify = tf.get_variable('b_classify', [1], initializer=self.const_initializer)
                out = tf.matmul(fea_label_plus_b, w_classify)
                # out = tf.add(wz_b_plus, b_classify)
                # out: [batch_size, 1]
        return tf.squeeze(out)
        #return tf.nn.relu(wz_b_plus)
        #return wz_b_plus
        #return tf.nn.softmax(out, -1)
        #return tf.nn.softmax(tf.nn.relu(wz_b_plus), -1)

    def build_model(self):
        # x: [batch_size, self.max_seq_len]
        # y: [batch_size]
        # label_embedding_id: [batch_size]
        x = tf.nn.embedding_lookup(self.word_embedding, self.x)
        # x: [batch_size, self.max_seq_len, word_embedding_dim]
        label_embeddings = tf.nn.embedding_lookup(self.label_embedding, self.label_embedding_id)
        # label_embeddings: [batch_size, label_embedding_dim]
        x_expand = tf.expand_dims(x, axis=-1)
        y = self.y
        # dropout
        # TODO
        #x_expand = tf.nn.dropout(x_expand, keep_prob=0.25)
        conv_outputs = []
        conv_atten_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('convolution-pooling-{0}'.format(filter_size)) as name_scope:
                # ============= convolution ============
                filter = tf.get_variable('filter-{0}'.format(filter_size),
                                         [filter_size, self.word_embedding_dim, 1, self.num_filters],
                                         initializer=self.weight_initializer)
                conv = tf.nn.conv2d(x_expand, filter, strides=[1,1,1,1], padding='VALID', name='conv')
                b = tf.get_variable('b-{0}'.format(filter_size), [self.num_filters])
                conv_b = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')
                # conv_b: [batch_size, seqence_length-filter_size+1, 1, num_filters]
                # ============= max pooling for x-embedding =========
                pool_emb = tf.nn.max_pool(conv_b, ksize=[1, self.max_seq_len-filter_size+1, 1, 1],
                                          strides=[1, 1, 1, 1], padding='VALID', name='max-pooling')
                # pool_emb: [batch_size, 1, 1, num_filters]
                conv_outputs.append(tf.squeeze(pool_emb, [1, 2]))
                # ============= dynamic max pooling =================
                pool_size = (self.max_seq_len - filter_size + 1) // self.pooling_units
                pool_out = tf.nn.max_pool(conv_b, ksize=[1, pool_size, 1, 1],
                                          strides=[1, pool_size, 1, 1], padding='VALID', name='dynamic-max-pooling')
                # pool_out: [batch_size, pooling_units, 1, num_filters]
                # ============= attention ===============
                pool_squeeze = tf.squeeze(pool_out, [-2])
                # pool_squeeze: [batch_size, pooling_units, num_filters]
                print [None, self.pooling_units, self.num_filters]
                print pool_squeeze.get_shape().as_list()
                num_hiddens = (self.max_seq_len - filter_size + 1) // pool_size
                print num_hiddens
                l_feature = self.attention_layer(pool_squeeze, label_embeddings, num_hiddens, self.num_filters, self.label_embedding_dim, name_scope=name_scope)
                # l_feature: [batch_size, num_filters]
                conv_atten_outputs.append(l_feature)
        x_emb = tf.concat(conv_outputs, -1)
        all_features = tf.concat(conv_atten_outputs, -1)
        # dropout
        with tf.name_scope('dropout'):
            fea_dropout = tf.nn.dropout(all_features, keep_prob=self.dropout_keep_prob)
        with tf.name_scope('output'):
            fea_dim = fea_dropout.get_shape().as_list()[-1]
            y_ = self.classification_layer(fea_dropout, label_embeddings, fea_dim, self.label_embedding_dim)
        # loss
        # y: [batch_size]
        #loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_))
        loss = tf.reduce_sum(
            tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_), self.label_prop)
        )
        # if self.use_propensity:
        #     loss = tf.losses.sigmoid_cross_entropy(y, y_, weights=tf.expand_dims(self.label_prop, -1))
        # else:
        #     loss = tf.losses.sigmoid_cross_entropy(y, y_)
        return x_emb, tf.sigmoid(y_), loss




