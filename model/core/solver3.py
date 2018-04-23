'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import time
import math
import numpy as np
import copy
import tensorflow as tf
from progressbar import *
from sklearn.neighbors import NearestNeighbors
# from biLSTM.preprocessing.preprocessing import batch_data, get_max_seq_len, construct_train_test_corpus, \
#     generate_labels_from_file, generate_label_pair_from_file
# from biLSTM.utils.io_utils import load_pickle, write_file, load_txt
from model.utils.op_utils import *
from model.utils.io_utils import load_pickle, dump_pickle


class ModelSolver3(object):
    def __init__(self, model, train_data, test_data, **kwargs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.if_use_seq_len = kwargs.pop('if_use_seq_len', 0)
        self.if_output_all_labels = kwargs.pop('if_output_all_labels', 0)
        self.show_batches = kwargs.pop('show_batches', 20)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.batch_pid_size = kwargs.pop('batch_pid_size', 4)
        self.alpha = kwargs.pop('alpha', 0.2)
        self.learning_rate = kwargs.pop('learning_rate', 0.0001)
        self.g_learning_rate = kwargs.pop('learning_rate', 0.0001)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.model_path = kwargs.pop('model_path', './model/')
        self.log_path = kwargs.pop('log_path', './log/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_path = kwargs.pop('test_path', None)
        self.use_graph = kwargs.pop('use_graph', 0)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer


    def train(self, output_file_path):
        o_file = open(output_file_path, 'w')
        train_loader = self.train_data
        test_loader = self.test_data
        # build_model
        _, y_, loss = self.model.build_model()
        # train op
        with tf.name_scope('optimizer'):
            # ========== loss
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        tf.get_variable_scope().reuse_variables()
        # set upper limit of used gpu memory
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            if self.pretrained_model is not None:
                print 'Start training with pretrained model...'
                pretrained_model_path = self.model_path + self.pretrained_model
                saver.restore(sess, pretrained_model_path)
            # ============== begin training ===================
            for e in xrange(self.n_epochs):
                print '========== begin epoch %d ===========' % e
                curr_loss = 0
                val_loss = 0
                # '''
                # ------------- train ----------------
                num_train_points = len(train_loader.train_pids)
                train_pid_batches = xrange(int(math.ceil(num_train_points * 1.0 / self.batch_size)))
                print 'num of train batches:    %d' % len(train_pid_batches)
                widgets = ['Train: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=len(train_pid_batches)).start()
                for i in train_pid_batches:
                    pbar.update(i)
                    _, x_feature_id, x_feature_v, seq_l, y = train_loader.get_pid_x(train_loader.train_pids,
                                                                                    i*self.batch_size, (i+1)*self.batch_size)
                    x_feature_v = x_feature_v/np.linalg.norm(x_feature_v, 2, axis=-1, keepdims=True)
                    #x_feature_v += np.random.normal(0, 0.01, x_feature_v.shape)
                    if len(y) == 0:
                        continue
                    feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                 self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                 self.model.y: np.array(y, dtype=np.float32),
                                 self.model.seqlen: np.array(seq_l, dtype=np.int32)
                                 }
                    _, l_ = sess.run([train_op, loss], feed_dict)
                    curr_loss += l_
                pbar.finish()
                # -------------- validate -------------
                num_val_points = len(train_loader.val_pids)
                val_pid_batches = xrange(int(math.ceil(num_val_points*1.0 / self.batch_size)))
                print 'num of validate pid batches: %d' % len(val_pid_batches)
                pre_pid_label_prop = {}
                tar_pid_label_prop = {}
                widgets = ['Validate: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=len(val_pid_batches)).start()
                for i in val_pid_batches:
                    pbar.update(i)
                    batch_pid, x_feature_id, x_feature_v, seq_l, y = train_loader.get_pid_x(train_loader.val_pids,
                                                                                            i*self.batch_size, (i+1)*self.batch_size)
                    x_feature_v = x_feature_v / np.linalg.norm(x_feature_v, 2, axis=-1, keepdims=True)
                    feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                 self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                 self.model.y: np.array(y),
                                 self.model.seqlen: np.array(seq_l)
                                 }
                    y_p, l_ = sess.run([y_, loss], feed_dict)
                    val_loss += l_
                    # prediction
                    for p_i in xrange(len(batch_pid)):
                        pid = batch_pid[p_i]
                        pre_label_index = np.argsort(-np.array(y_p[p_i]))[:5]
                        pre_pid_label_prop[pid] = [y[ind]*(train_loader.label_prop[ind]) for ind in pre_label_index]
                        tar_pid_label_prop[pid] = [train_loader.label_prop[q] for q in train_loader.label_data[pid]]
                pbar.finish()
                val_results = results_for_prop_vector(tar_pid_label_prop, pre_pid_label_prop)
                # reset train_loader
                train_loader.reset_data()
                # ====== output loss ======
                w_text = 'at epoch %d, train loss is %f \n' % (e, curr_loss/len(train_pid_batches))
                print w_text
                o_file.write(w_text)
                w_text = 'at epoch %d, val loss is %f \n' % (e, val_loss/len(val_pid_batches))
                print w_text
                o_file.write(w_text)
                w_text = 'at epoch %d, val_results: ' % e
                w_text = w_text + str(val_results)
                print w_text
                o_file.write(w_text)
                # ====== save model ========
                save_name = self.model_path + 'model'
                saver.save(sess, save_name, global_step=e+1)
                print 'model-%s saved.' % (e+1)
                # '''
                # ----------------- test ---------------------
                if e % 2 == 0:
                    print '=============== test ================'
                    test_loss = 0
                    num_test_points = len(test_loader.pids)
                    test_pid_batches = xrange(int(math.ceil(num_test_points * 1.0 / self.batch_size)))
                    print 'num of test pid batches: %d' % len(test_pid_batches)
                    pre_pid_label_prop = {}
                    tar_pid_label_prop = {}
                    widgets = ['Validate: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                    pbar = ProgressBar(widgets=widgets, maxval=len(test_pid_batches)).start()
                    for i in test_pid_batches:
                        pbar.update(i)
                        batch_pid, x_feature_id, x_feature_v, seq_l, y = test_loader.get_pid_x(test_loader.pids,
                                                                                                i * self.batch_size, (
                                                                                                i + 1) * self.batch_size)
                        x_feature_v = x_feature_v / np.linalg.norm(x_feature_v, 2, axis=-1, keepdims=True)
                        feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                     self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                     self.model.y: np.array(y),
                                     self.model.seqlen: np.array(seq_l)
                                     }
                        y_p, l_ = sess.run([y_, loss], feed_dict)
                        test_loss += l_
                        # prediction
                        for p_i in xrange(len(batch_pid)):
                            pid = batch_pid[p_i]
                            pre_label_index = np.argsort(-np.array(y_p[p_i]))[:5]
                            pre_pid_label_prop[pid] = [y[ind] * (test_loader.label_prop[ind]) for ind in
                                                       pre_label_index]
                            tar_pid_label_prop[pid] = [test_loader.label_prop[q] for q in test_loader.label_data[pid]]
                    pbar.finish()
                    test_results = results_for_prop_vector(tar_pid_label_prop, pre_pid_label_prop)
                    w_text = 'at epoch %d, test loss is %f \n' % (e, test_loss/len(test_pid_batches))
                    print w_text
                    o_file.write(w_text)
                    p1_txt = 'prec_wt@1: %f \n' % test_results[0]
                    p3_txt = 'prec_wt@3: %f \n' % test_results[1]
                    p5_txt = 'prec_wt@5: %f \n' % test_results[2]
                    ndcg1_txt = 'ndcg_wt@1: %f \n' % test_results[3]
                    ndcg3_txt = 'ndcg_wt@3: %f \n' % test_results[4]
                    ndcg5_txt = 'ndcg_wt@5: %f \n' % test_results[5]
                    o_file.write(p1_txt + p3_txt + p5_txt + ndcg1_txt + ndcg3_txt + ndcg5_txt)
                    print p1_txt + p3_txt + p5_txt + ndcg1_txt + ndcg3_txt + ndcg5_txt
            # save model
            save_name = self.model_path + 'model_final'
            saver.save(sess, save_name)
            print 'final model saved.'
            o_file.close()
