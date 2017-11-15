'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import tensorflow as tf

class biLSTM(object):
    def __init__(self, seq_max_len, input_dim, num_label, num_hidden, num_classify_hidden, label_embeddings, batch_size):
        self.num_hidden = num_hidden
        self.seq_max_len = seq_max_len
        self.input_dim = input_dim
        self.num_label = num_label
        self.num_classify_hidden = num_classify_hidden
        self.label_embeddings = tf.cast(label_embeddings, tf.float32)
        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.seq_max_len, self.input_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.num_label])
        self.seqlen = tf.placeholder(tf.int32, [None])

    def attention_layer(self, hidden_states, label_embedding, num_hidden, num_label_embedding):
        # attention: a*W*b
        with tf.variable_scope('att_layer'):
            w = tf.get_variable('w', [num_hidden, num_label_embedding], initializer=self.weight_initializer)
            #b = tf.get_variable('b', [])
            s = tf.matmul(tf.matmul(hidden_states, w), label_embedding)
            #s = tf.softmax(s, 0)
            s = tf.expand_dim(tf.softmax(s))
            return tf.reduce_sum(tf.multiply(s, hidden_states), 0)

    def attention_layer_all(self, hidden_states, label_embeddings, num_hidden, num_label_embedding, reuse):
        with tf.variable_scope('att_layer', reuse=reuse):
            w = tf.get_variable('w', [num_hidden, num_label_embedding], initializer=self.weight_initializer)
            # hidden_states: [seq_len, num_hidden]
            # label_embeddings: [num_label, num_label_embedding]
            s = tf.matmul(tf.matmul(hidden_states, w), tf.transpose(label_embeddings))
            # s: [seq_len, num_label]
            s = tf.nn.softmax(s, 0)
            # s_expand: [num_label, seq_len, 1]
            s_expand = tf.expand_dims(tf.transpose(s), axis=-1)
            # s_hidden: [num_label, seq_len, num_hidden]
            s_hidden = tf.multiply(s_expand, hidden_states)
            return tf.reduce_sum(s_hidden, axis=1)

    def classification(self, hidden_states, label_embeddings, num_hidden, num_label_embedding, reuse):
        # z: [num_label, num_hidden] (num_z = num_hidden)
        # label_embeddings: [num_label, num_label_embedding]
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
            # z_label_plus: [num_label, num_classify_hidden]
            #
            w_classify = tf.get_variable('w_classify', [self.num_classify_hidden, 2], initializer=self.weight_initializer)
            b_classify = tf.get_variable('b_classify', [2], initializer=self.const_initializer)
            wz_b_plus = tf.matmul(z_label_plus, w_classify) + b_classify
            # wz_b_plus: [num_label, 2]
            return tf.nn.softmax(tf.nn.relu(wz_b_plus), -1)


    def build_model(self):
        # x: [batch_size, self.seq_max_len, self.input_dim]
        # y: [batch_size, self.num_label]
        x = self.x
        y = self.y
        # transform y to [batch_size, self.num_label, 2]
        y = tf.expand_dims(y, -1)
        y = tf.concat([1-y, y], axis=1)

        #batch_size = tf.shape(x)[0]
        # batch_size = x.get_shape().as_list()[0]

        # ----------- biLSTM ------------
        # activation: tanh(default)
        fw_lstm = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
        bw_lstm = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)

        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([fw_lstm], [bw_lstm], x, dtype=tf.float32, sequence_length=self.seqlen)
        outputs = tf.stack(outputs)
        print('outputs_shape : ', outputs.get_shape().as_list())
        # transpose the output back to [batch_size, n_step, num_hidden)
        # outputs = tf.transpose(outputs, [1, 0, 2])

        #index = tf.range(0, batch_size)*self.seq_max_len + (self.seqlen - 1)
        # Indexing
        # outputs = tf.gather(tf.reshape(outputs, [-1, self.num_hidden]), index)
        # print('after indexing, outputs_shape : ', outputs.get_shape().as_list())
        # ------------ attention and classification --------------
        num_label_embedding = self.label_embeddings.shape[-1]
        y_ = []
        #print 'clssify:'
        for i in range(self.batch_size):
            #hidden_states = outputs[i, 0:self.seqlen[i], :]
            #print i
            y_.append(self.classification(outputs[i, 0:self.seqlen[i], :],
                                          self.label_embeddings,
                                          2*self.num_hidden,
                                          num_label_embedding,
                                          reuse=(i!=0)))
        y_ = tf.stack(y_)
        # predict labels
        y_labels = tf.argmax(y_, axis=-1)
        y_labels_prob = y_[:,:,-1]
        print y_labels_prob.get_shape().as_list()
        # calculate loss
        y_tar = tf.reshape(y, [-1, 2])
        y_pre = tf.reshape(y_, [-1, 2])
        loss = tf.losses.sigmoid_cross_entropy(y_tar, y_pre)
        return y_labels_prob, loss




