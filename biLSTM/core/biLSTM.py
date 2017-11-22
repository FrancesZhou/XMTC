'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import tensorflow as tf

class biLSTM(object):
    def __init__(self, max_seq_len, input_dim, num_label_embedding, num_hidden, num_classify_hidden):
        self.num_hidden = num_hidden
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        #self.num_labels = num_labels
        self.num_label_embedding = num_label_embedding
        self.num_classify_hidden = num_classify_hidden
        #self.label_embeddings = tf.cast(label_embeddings, tf.float32)
        #self.label_embeddings = tf.Variable(tf.cast(label_embeddings, tf.float32))
        #self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder(tf.float32, [None, self.max_seq_len, self.input_dim])
        self.y = tf.placeholder(tf.float32, [None, 2])
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.label_embeddings = tf.placeholder(tf.float32, [None, self.num_label_embedding])

    def attention_layer(self, hidden_states, label_embedding, num_hidden, num_label_embedding):
        # attention: a*W*b
        with tf.variable_scope('att_layer'):
            w = tf.get_variable('w', [num_hidden, num_label_embedding], initializer=self.weight_initializer)
            s = tf.matmul(tf.matmul(hidden_states, w), label_embedding)
            s = tf.expand_dim(tf.softmax(s))
            return tf.reduce_sum(tf.multiply(s, hidden_states), 0)

    def attention_layer_all(self, hidden_states, label_embeddings, num_hidden, num_label_embedding, reuse):
        # for one label, num_labels = 1
        with tf.variable_scope('att_layer', reuse=reuse):
            w = tf.get_variable('w', [num_hidden, num_label_embedding], initializer=self.weight_initializer)
            # hidden_states: [seq_len, num_hidden]
            # label_embeddings: [num_labels, num_label_embedding]
            # score: h*W*l
            s = tf.matmul(tf.matmul(hidden_states, w), tf.transpose(label_embeddings))
            # s: [seq_len, num_labels]
            s = tf.nn.softmax(s, 0)
            # s_expand: [num_labels, seq_len, 1]
            s_expand = tf.expand_dims(tf.transpose(s), axis=-1)
            # s_hidden: [num_labels, seq_len, num_hidden]
            s_hidden = tf.multiply(s_expand, hidden_states)
            return tf.reduce_sum(s_hidden, axis=1)

    def classification(self, hidden_states, label_embeddings, num_hidden, num_label_embedding, reuse):
        # z: [num_labels, num_hidden] (num_z = num_hidden)
        # label_embeddings: [num_labels, num_label_embedding]
        # --------- attention -------------
        z = self.attention_layer_all(hidden_states, label_embeddings, num_hidden, num_label_embedding, reuse=reuse)
        # --------- classification --------
        with tf.variable_scope('classify_layer', reuse=reuse):
            with tf.variable_scope('z'):
                w = tf.get_variable('w', [num_hidden, self.num_classify_hidden], initializer=self.weight_initializer)
                z_att = tf.matmul(z, w)
            with tf.variable_scope('label'):
                w = tf.get_variable('w', [num_label_embedding, self.num_classify_hidden], initializer=self.weight_initializer)
                label_att = tf.matmul(label_embeddings, w)
            b = tf.get_variable('b', [self.num_classify_hidden], initializer=self.const_initializer)
            z_label_plus = tf.nn.relu(z_att + label_att) + b
            # z_label_plus: [num_labels, num_classify_hidden]
            #
            w_classify = tf.get_variable('w_classify', [self.num_classify_hidden, 2], initializer=self.weight_initializer)
            b_classify = tf.get_variable('b_classify', [2], initializer=self.const_initializer)
            wz_b_plus = tf.matmul(z_label_plus, w_classify) + b_classify
            # wz_b_plus: [num_labels, 2]
            return tf.nn.softmax(tf.nn.relu(wz_b_plus), -1)


    def build_model(self):
        # x: [batch_size, self.max_seq_len, self.input_dim]
        # y: [batch_size, 2]
        x = self.x
        y = self.y
        #batch_size = tf.shape(x)[0]
        batch_size = x.get_shape().as_list()[0]

        # ----------- biLSTM ------------
        # activation: tanh(default)
        fw_lstm = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
        bw_lstm = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([fw_lstm], [bw_lstm], x, dtype=tf.float32, sequence_length=self.seqlen)
        outputs = tf.stack(outputs)
        print('outputs_shape : ', outputs.get_shape().as_list())
        # output: [batch_size, n_step, num_hidden)
        # index = tf.range(0, batch_size)*self.max_seq_len + (self.seqlen - 1)
        # Indexing
        # outputs = tf.gather(tf.reshape(outputs, [-1, self.num_hidden]), index)
        # print('after indexing, outputs_shape : ', outputs.get_shape().as_list())
        # ------------ attention and classification --------------
        #num_label_embedding = self.label_embeddings.shape[-1]
        y_ = []
        for i in range(batch_size):
            #hidden_states = outputs[i, 0:self.seqlen[i], :]
            #print i
            y_i = self.classification(outputs[i, 0:self.seqlen[i], :],
                                          self.label_embeddings[i],
                                          2*self.num_hidden,
                                          self.num_label_embedding,
                                          reuse=(i!=0))
            print tf.shape(tf.squeeze(y_i))
            y_.append(tf.squeeze(y_i))
        y_ = tf.stack(y_)
        # predict labels
        # y_labels = tf.argmax(y_, axis=-1)
        # y_labels_prob = y_[:,:,-1]
        # print y_labels_prob.get_shape().as_list()
        # calculate loss
        # y_tar = tf.reshape(y, [-1, 2])
        # y_pre = tf.reshape(y_, [-1, 2])
        y_pre = y_
        y_tar = y
        loss = tf.losses.sigmoid_cross_entropy(y_tar, y_pre)
        return y_pre[:, 1], loss




