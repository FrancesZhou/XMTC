'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import time
import numpy as np
import tensorflow as tf
# from biLSTM.preprocessing.preprocessing import batch_data, get_max_seq_len, construct_train_test_corpus, \
#     generate_labels_from_file, generate_label_pair_from_file
# from biLSTM.utils.io_utils import load_pickle, write_file, load_txt
from biLSTM.utils.op_utils import precision, precision_for_all


class ModelSolver(object):
    def __init__(self, model, train_data, test_data, **kwargs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.learning_rate = kwargs.pop('learning_rate', 0.000001)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        # self.model_path = kwargs.pop('model_path', './model/')
        # self.save_every = kwargs.pop('save_every', 1)
        # self.log_path = kwargs.pop('log_path', './log/')
        # self.pretrained_model = kwargs.pop('pretrained_model', None)
        # self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)
        # if not os.path.exists(self.log_path):
        #     os.makedirs(self.log_path)
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

    def train(self):
        train_loader = self.train_data
        test_loader = self.test_data

        # build_model
        y_, loss = self.model.build_model()
        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        tf.get_variable_scope().reuse_variables()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for e in range(6):
                # for e in range(self.n_epochs):
                curr_loss = 0
                #for i in range(200):
                i = 0
                #for i in range(10):
                '''
                while not train_loader.end_of_data:
                    if i % 10 == 0:
                        print i
                    _, _, x, y, seq_l, label_emb = train_loader.next_batch()
                    if len(x) < self.batch_size:
                        train_loader.reset_data()
                        break
                    feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                 self.model.seqlen: np.array(seq_l), self.model.label_embeddings: label_emb}
                    _, l_ = sess.run([train_op, loss], feed_dict)
                    curr_loss += l_
                    i += 1
                else:
                    train_loader.reset_data()
                print('at epoch ' + str(e) + ', train loss is ' + str(curr_loss))
                '''

                # ----------------- test ---------------------
                if e in [1, 3, 5]:
                    print '=============== test ================'
                    val_loss = 0
                    pred_pid_label = dict.fromkeys(test_loader.label_data.keys(), [])
                    pred_pid_score = dict.fromkeys(test_loader.label_data.keys(), [])
                    #for i in range(200):
                    i = 0
                    #for i in range(10):
                    while not test_loader.end_of_data:
                        if i % 10 == 0:
                            print i
                        batch_pid, batch_label, x, y, seq_l, label_emb = test_loader.next_batch()
                        if len(batch_pid) < self.batch_size:
                            test_loader.reset_data()
                            break
                        feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                     self.model.seqlen: np.array(seq_l), self.model.label_embeddings: label_emb}
                        y_p, l_ = sess.run([y_, loss], feed_dict)
                        val_loss += l_
                        #batch_pre = precision(y_p, y, indices)
                        #metric.append(batch_pre)
                        i += 1
                        # get all predictions
                        for j in range(len(batch_pid)):
                            pred_pid_label[batch_pid[j]].append(batch_label[j])
                            pred_pid_score[batch_pid[j]].append(y_p[j])
                    else:
                        test_loader.reset_data()
                    # mean_metric = np.mean(metric, axis=0)
                    mean_metric = precision_for_all(test_loader.label_data, pred_pid_label, pred_pid_score)
                    print 'at epoch' + str(e) + ', test loss is ' + str(val_loss)
                    print 'precision@1: ' + str(mean_metric[0])
                    print 'precision@3: ' + str(mean_metric[1])
                    print 'precision@5: ' + str(mean_metric[2])
                    print 'ndcg@1: ' + str(mean_metric[3])
                    print 'ndcg@3: ' + str(mean_metric[4])
                    print 'ndcg@5: ' + str(mean_metric[5])

                    # def train(self):
                    #     x = self.train_data['x']
                    #     y = self.train_data['y']
                    #     l = self.train_data['l']
                    #
                    #     x_test = self.test_data['x']
                    #     y_test = self.test_data['y']
                    #     l_test = self.test_data['l']
                    #
                    #     y_test_concate = np.concatenate(y_test, axis=0)
                    #
                    #     # build_model
                    #     y_, loss = self.model.build_model()
                    #
                    #     # train op
                    #     with tf.name_scope('optimizer'):
                    #         optimizer = self.optimizer(learning_rate=self.learning_rate)
                    #         grads = tf.gradients(loss, tf.trainable_variables())
                    #         grads_and_vars = list(zip(grads, tf.trainable_variables()))
                    #         train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
                    #
                    #     tf.get_variable_scope().reuse_variables()
                    #
                    #     # summary op
                    #     # tf.summary.scalar('batch_loss', loss)
                    #     # for var in tf.trainable_variables():
                    #     #     tf.summary.histogram(var.op.name, var)
                    #     # for grad, var in grads_and_vars:
                    #     #     tf.summary.histogram(var.op.name + '/gradient', grad)
                    #     # summary_op = tf.summary.merge_all()
                    #
                    #     with tf.Session() as sess:
                    #         tf.global_variables_initializer().run()
                    #         #summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
                    #         #saver = tf.train.Saver(tf.global_variables())
                    #         # if self.pretrained_model is not None:
                    #         #     print "Start training with pretrained model..."
                    #         #     saver.restore(sess, self.pretrained_model)
                    #
                    #         #start_t = time.time()
                    #         for e in range(self.n_epochs):
                    #             curr_loss = 0
                    #             for i in range(len(x)):
                    #                 feed_dict = {self.model.x: np.array(x[i]), self.model.y: np.array(y[i]), self.model.seqlen: np.array(l[i])}
                    #                 _, l_ = sess.run([train_op, loss], feed_dict)
                    #                 curr_loss += l_
                    #
                    #                 # # write summary for tensorboard visualization
                    #                 # if i % 100 == 0:
                    #                 #     print("at epoch " + str(e) + ', ' + str(i))
                    #                 #     summary = sess.run(summary_op, feed_dict)
                    #                 #     summary_writer.add_summary(summary, e * len(x) + i)
                    #             print('at epoch ' + str(e) + ', train loss is ' + str(curr_loss))
                    #
                    #             # --- test ---
                    #             if e%2 == 0:
                    #                 val_loss = 0
                    #                 y_prob = []
                    #                 for i in range(len(x_test)):
                    #                     feed_dict = {self.model.x: np.array(x_test[i]), self.model.y: np.array(y_test[i]),
                    #                                  self.model.seqlen: np.array(l_test[i])}
                    #                     y_p, l_ = sess.run([y_, loss], feed_dict)
                    #                     val_loss += l_
                    #                     y_prob.append(y_p)
                    #                 y_prob = np.concatenate(y_prob, axis=0)
                    #                 precision_1 = self.precision(y_prob, y_test_concate, 1)
                    #                 precision_3 = self.precision(y_prob, y_test_concate, 3)
                    #                 precision_5 = self.precision(y_prob, y_test_concate, 5)
                    #                 print 'at epoch'+str(e)+', test loss is '+str(val_loss)
                    #                 print 'precision@1: ' + str(precision_1)
                    #                 print 'precision@3: ' + str(precision_3)
                    #                 print 'precision@5: ' + str(precision_5)
                    #                 #print "elapsed time: ", time.time() - start_t
