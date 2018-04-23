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
from model.utils.op_utils import ndcg_at_k, precision_for_pre_label, precision_for_all, results_for_score_vector
from model.utils.io_utils import load_pickle, dump_pickle


class ModelSolver2(object):
    def __init__(self, model, train_data, test_data, feature_processor, graph_data=None, **kwargs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.feature_processor = feature_processor
        self.graph_data = graph_data
        self.train_x_emb = {}
        self.test_x_emb = {}
        self.test_unique_candidate_label = {}
        self.test_all_candidate_label = {}
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
        graph_loader = self.graph_data
        # build_model
        _, _, word_grads, pre_loss, pre_g_loss = self.model.pre_build_model()
        _, y_, loss, g_loss = self.model.build_model()
        # train op
        with tf.name_scope('optimizer'):
            # ========= pre_loss
            pre_optimizer = self.optimizer(learning_rate=self.learning_rate)
            #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            for grad_var in grads_and_vars:
                g, v = grad_var
                print g
                print v
            pre_train_op = pre_optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            # summary op
            #tf.summary.scalar('batch_loss', pre_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            for grad, var in grads_and_vars:
                if grad is not None:
                    tf.summary.histogram(var.op.name+'/gradient', grad)
                else:
                    tf.summary.histogram(var.op.name+'/gradient', tf.zeros([1], tf.int32))
            summary_op = tf.summary.merge_all()
            if self.use_graph:
                pre_g_train_op = pre_optimizer.minimize(g_loss, global_step=tf.train.get_global_step())
            #
            # ========== loss
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            # ========== graph
            if self.use_graph:
                g_optimizer = self.optimizer(learning_rate=self.g_learning_rate)
                g_train_op = g_optimizer.minimize(g_loss, global_step=tf.train.get_global_step())
        tf.get_variable_scope().reuse_variables()
        # set upper limit of used gpu memory
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(tf.global_variables())
            if self.pretrained_model is not None:
                print 'Start training with pretrained model...'
                pretrained_model_path = self.model_path + self.pretrained_model
                saver.restore(sess, pretrained_model_path)
            # ============== pretrain ======================
            print '------------- begin pretrain --------------'
            for e in xrange(10):
                curr_loss = 0
                curr_g_loss = 0
                num_train_batches = len(train_loader.all_labels)
                train_batches = np.arange(num_train_batches)
                np.random.shuffle(train_batches)
                widgets = ['Pretrain: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=num_train_batches).start()
                for i_ in xrange(num_train_batches):
                    i = train_batches[i_]
                    pbar.update(i_)
                    label_i = train_loader.all_labels[i]
                    x_feature_id, x_feature_v, y, seq_l, label_emb, label_prop \
                        = train_loader.next_batch(label_i)
                    x_feature_v = x_feature_v/np.linalg.norm(x_feature_v, 2, axis=-1, keepdims=True)
                    x_feature_v += np.random.normal(0, 0.01, x_feature_v.shape)
                    if len(y) == 0:
                        continue
                    if self.use_graph:
                        gx1, gx2, gy = graph_loader.gen_graph_context()
                    else:
                        gx1, gx2, gy = [0], [0], [0]
                    if self.if_use_seq_len:
                        feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                     self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                     self.model.y: np.array(y, dtype=np.float32),
                                     self.model.seqlen: np.array(seq_l, dtype=np.int32),
                                     self.model.label_embedding_id: np.array(label_emb, dtype=np.int32),
                                     self.model.label_prop: np.array(label_prop, dtype=np.float32),
                                     self.model.gl1: np.array(gx1, dtype=np.int32),
                                     self.model.gl2: np.array(gx2, dtype=np.int32),
                                     self.model.gy: np.array(gy, dtype=np.float32)
                                     }
                    _, l_, w_grads = sess.run([pre_train_op, pre_loss, word_grads], feed_dict)
                    self.feature_processor.set_active_feature_grads(label_i, w_grads)
                    curr_loss += l_
                    if self.use_graph:
                        _, gl_ = sess.run([pre_g_train_op, pre_g_loss], feed_dict)
                        curr_g_loss += gl_
                    # write summary for tensorboard visualization
                    #if i%100 == 0:
                    #    summary = sess.run(summary_op, feed_dict)
                    #    summary_writer.add_summary(summary, i_*len(y) + i)
                pbar.finish()
                w_text = 'at epoch %d, g_loss = %f , train loss is %f \n' % \
                         (e, curr_g_loss / num_train_batches, curr_loss / num_train_batches)
                print w_text
                #pbar.finish()
            # ===== set active feature ids for each label
            self.feature_processor.set_active_feature_id()
            # ============== begin training ===================
            for e in xrange(self.n_epochs):
                print '========== begin epoch %d ===========' % e
                curr_loss = 0
                curr_g_loss = 0
                val_loss = 0
                # '''
                # ------------- train ----------------
                num_train_batches = len(train_loader.all_labels)
                print 'num of train batches:    %d' % num_train_batches
                train_batches = np.arange(num_train_batches)
                np.random.shuffle(train_batches)
                widgets = ['Train: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=num_train_batches).start()
                for i_ in xrange(num_train_batches):
                    i = train_batches[i_]
                    pbar.update(i_)
                    label_i = train_loader.all_labels[i]
                    x_feature_id, x_feature_v, y, seq_l, label_emb, label_prop \
                        = train_loader.next_batch(label_i)
                    x_feature_v = x_feature_v/np.linalg.norm(x_feature_v, 2, axis=-1, keepdims=True)
                    x_feature_v += np.random.normal(0, 0.01, x_feature_v.shape)
                    if len(y) == 0:
                        continue
                    if self.use_graph:
                        gx1, gx2, gy = graph_loader.gen_graph_context()
                    else:
                        gx1, gx2, gy = [0], [0], [0]
                    lbl_active_fea_id = np.tile(self.feature_processor.label_active_feature_ids[label_i], (len(y), 1))
                    if self.if_use_seq_len:
                        feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                     self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                     self.model.y: np.array(y, dtype=np.float32),
                                     self.model.seqlen: np.array(seq_l, dtype=np.int32),
                                     self.model.label_embedding_id: np.array(label_emb, dtype=np.int32),
                                     self.model.label_prop: np.array(label_prop, dtype=np.float32),
                                     self.model.gl1: np.array(gx1, dtype=np.int32),
                                     self.model.gl2: np.array(gx2, dtype=np.int32),
                                     self.model.gy: np.array(gy, dtype=np.float32),
                                     self.model.label_active_feature: np.array(lbl_active_fea_id, dtype=np.int32)
                                     }
                    _, l_ = sess.run([train_op, loss], feed_dict)
                    curr_loss += l_
                    if self.use_graph:
                        _, gl_ = sess.run([g_train_op, g_loss], feed_dict)
                        curr_g_loss += gl_
                pbar.finish()
                # -------------- validate -------------
                num_val_points = len(train_loader.val_pid_label_y)
                val_pid_batches = xrange(int(math.ceil(num_val_points*1.0 / self.batch_size)))
                print 'num of validate pid batches: %d' % len(val_pid_batches)
                pre_pid_prop = {}
                pre_pid_score = {}
                tar_pid_y = {}
                tar_pid_true_label_prop = {}
                widgets = ['Validate: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=num_val_points).start()
                for i in val_pid_batches:
                    pbar.update(i)
                    lbl_active_fea_id, batch_pid, x_feature_id, x_feature_v, y, seq_l, label_emb, label_prop, count_score \
                        = train_loader.get_val_batch(num_val_points, i*self.batch_size, (i+1)*self.batch_size)
                    if self.if_use_seq_len:
                        feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                     self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                     self.model.y: np.array(y),
                                     self.model.seqlen: np.array(seq_l),
                                     self.model.label_embedding_id: np.array(label_emb, dtype=int),
                                     self.model.label_prop: np.array(label_prop),
                                     self.model.label_active_feature: np.array(lbl_active_fea_id, dtype=np.int32)
                                     }
                    y_p, l_ = sess.run([y_, loss], feed_dict)
                    val_loss += l_
                    # prediction
                    for p_i in xrange(len(batch_pid)):
                        pid = batch_pid[p_i]
                        try:
                            tar_pid_y[pid].append(y[p_i])
                            pre_pid_score[pid].append(y_p[p_i])
                            #pre_pid_score[pid].append(np.multiply(
                            #    np.power(y_p[p_i], self.alpha), np.power(count_score[p_i], 1-self.alpha)
                            #))
                            pre_pid_prop[pid].append(label_prop[p_i])
                        except KeyError:
                            tar_pid_y[pid] = [y[p_i]]
                            pre_pid_score[pid] = [y_p[p_i]]
                            #pre_pid_score[pid] = [np.multiply(
                            #    np.power(y_p[p_i], self.alpha), np.power(count_score[p_i], 1-self.alpha)
                            #)]
                            pre_pid_prop[pid] = [label_prop[p_i]]
                    for pid in np.unique(batch_pid):
                        tar_pid_true_label_prop[pid] = [train_loader.label_prop[q] for q in train_loader.label_data[pid]]
                pbar.finish()
                val_results = results_for_score_vector(tar_pid_true_label_prop, tar_pid_y, pre_pid_score, pre_pid_prop)
                # reset train_loader
                train_loader.reset_data()
                # ====== output loss ======
                w_text = 'at epoch %d, g_loss = %f , train loss is %f \n' % (e, curr_g_loss/num_train_batches, curr_loss/num_train_batches)
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
                    pre_pid_prop = {}
                    pre_pid_score = {}
                    tar_pid_y = {}
                    tar_pid_true_label_prop = {}
                    num_test_points = len(test_loader.pid_label_y)
                    test_pid_batches = xrange(int(math.ceil(num_test_points * 1.0 / self.batch_size)))
                    print 'num of test pid batches: %d' % len(test_pid_batches)
                    widgets = ['Test: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                    pbar = ProgressBar(widgets=widgets, maxval=num_test_points).start()
                    for i in test_pid_batches:
                        pbar.update(i)
                        lbl_active_fea_id, batch_pid, x_feature_id, x_feature_v, y, seq_l, label_emb, label_prop, count_score = test_loader.get_batch(
                                num_test_points, i * self.batch_size, (i + 1) * self.batch_size)
                        if self.if_use_seq_len:
                            feed_dict = {self.model.x_feature_id: np.array(x_feature_id, dtype=np.int32),
                                         self.model.x_feature_v: np.array(x_feature_v, dtype=np.float32),
                                         self.model.y: np.array(y),
                                         self.model.seqlen: np.array(seq_l),
                                         self.model.label_embedding_id: np.array(label_emb, dtype=np.int32),
                                         self.model.label_prop: np.array(label_prop),
                                         self.model.label_active_feature: np.array(lbl_active_fea_id, dtype=np.int32)
                                         }
                        y_p, l_ = sess.run([y_, loss], feed_dict)
                        test_loss += l_
                        # get all predictions
                        # prediction
                        for p_i in xrange(len(batch_pid)):
                            pid = batch_pid[p_i]
                            try:
                                tar_pid_y[pid].append(y[p_i])
                                pre_pid_score[pid].append(y_p[p_i])
                                #pre_pid_score[pid].append(np.multiply(
                                #    np.power(y_p[p_i], self.alpha), np.power(count_score[p_i], 1 - self.alpha)
                                #))
                                pre_pid_prop[pid].append(label_prop[p_i])
                            except KeyError:
                                tar_pid_y[pid] = [y[p_i]]
                                pre_pid_score[pid] = [y_p[p_i]]
                                #pre_pid_score[pid] = [np.multiply(
                                #    np.power(y_p[p_i], self.alpha), np.power(count_score[p_i], 1 - self.alpha)
                                #)]
                                pre_pid_prop[pid] = [label_prop[p_i]]
                        for pid in np.unique(batch_pid):
                            tar_pid_true_label_prop[pid] = [test_loader.label_prop[q] for q in
                                                            test_loader.label_data[pid]]
                    pbar.finish()
                    test_results = results_for_score_vector(tar_pid_true_label_prop, tar_pid_y, pre_pid_score,
                                                                   pre_pid_prop)
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

    def test(self, trained_model_path, output_file_path, test_loader=None):
        o_file = open(output_file_path, 'w')
        if not test_loader:
            test_loader = self.test_data
        #test_loader.reset_data()
        # restore trained_model
        _, y_, loss = self.model.build_model()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            print 'load trained model...'
            model_name = trained_model_path
            saver.restore(sess, model_name)
            # -------------- test -------------
            print 'begin testing...'
            test_loss = 0
            if self.if_output_all_labels:
                pre_pid_score = {}
                tar_pid_label = {}
                k = 0
                batches = np.arange(math.ceil(len(test_loader.pids) * 1.0 / self.batch_size), dtype=int)
                # np.random.shuffle(batches)
                for i in batches:
                    if k % self.show_batches == 0:
                        print 'batch ' + str(i)
                    batch_pid, batch_x, batch_y = test_loader.get_pid_x(int(i * self.batch_size),
                                                                        int((i + 1) * self.batch_size))
                    try:
                        feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(batch_y)}
                    except:
                        print i
                    y_p, l_ = sess.run([y_, loss], feed_dict)
                    # print l_
                    test_loss += l_
                    k += 1
                    # get all predictions
                    for j in range(len(batch_pid)):
                        # tar_pid_label[batch_pid[j]] = np.squeeze(np.nonzero(batch_y[j]))
                        pre_pid_score[batch_pid[j]] = np.argpartition(-y_p[j], 5)[:5]
                        # pre_pid_score[batch_pid[j]] = heapq.nlargest
                mean_metric = precision_for_label_vector(test_loader.label_data, pre_pid_score)
            else:
                pre_pid_prop = {}
                pre_pid_score = {}
                tar_pid_y = {}
                tar_pid_true_label_prop = {}
                num_test_points = len(test_loader.pid_label_y)
                #self.batch_size = self.batch_pid_size
                #num_test_points = len(test_loader.pids)
                test_batches = xrange(int(math.ceil(num_test_points * 1.0 / self.batch_size)))
                print 'num of test batches: %d' % len(test_batches)
                #for i in xrange(len(self.pids)):
                for i in test_batches:
                    if i % self.show_batches == 0:
                        print 'batch ' + str(i)
                    batch_pid, x, y, seq_l, label_emb, label_prop, count_score = test_loader.get_batch(
                            num_test_points, i * self.batch_size, (i + 1) * self.batch_size)
                    if self.if_use_seq_len:
                        feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                     self.model.seqlen: np.array(seq_l),
                                     self.model.label_embedding_id: np.array(label_emb, dtype=int),
                                     self.model.label_prop: np.array(label_prop)
                                     }
                    else:
                        feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                     self.model.label_embedding_id: np.array(label_emb, dtype=int),
                                     self.model.label_prop: np.array(label_prop)
                                     }
                    y_p, l_ = sess.run([y_, loss], feed_dict)
                    test_loss += l_
                    #get all predictions
                    for p_i in xrange(len(batch_pid)):
                        pid = batch_pid[p_i]
                        try:
                            tar_pid_y[pid].append(y[p_i])
                            pre_pid_score[pid].append(np.multiply(np.power(y_p[p_i], self.alpha),
                                                                  np.power(count_score[p_i], 1 - self.alpha)))
                            pre_pid_prop[pid].append(label_prop[p_i])
                        except KeyError:
                            tar_pid_y[pid] = [y[p_i]]
                            pre_pid_score[pid] = [np.multiply(np.power(y_p[p_i], self.alpha),
                                                              np.power(count_score[p_i], 1 - self.alpha))]
                            pre_pid_prop[pid] = [label_prop[p_i]]
                    for pid in np.unique(batch_pid):
                        tar_pid_true_label_prop[pid] = [test_loader.label_prop[q] for q in
                                                        test_loader.label_data[pid]]
                    # for pid in batch_pid:
                    #     tar_pid_y[pid] = y
                    #     tar_pid_true_label_prop[pid] = [test_loader.label_prop[q] for q in test_loader.label_data[pid]]
                    #     pre_pid_score[pid] = np.multiply(np.power(y_p, 0.2), np.power(count_score, 0.8))
                    #     pre_pid_prop[pid] = label_prop
                test_results = results_for_score_vector(tar_pid_true_label_prop, tar_pid_y, pre_pid_score,
                                                        pre_pid_prop)
            w_text = 'test loss is %f \n' % test_loss
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

    def generate_x_embedding(self, trained_model_path):
        # generate candidate label subset via KNN using X-embeddings.
        # build model
        x_emb, y_, loss = self.model.build_model()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            print 'load trained model...'
            model_name = trained_model_path + 'model_final'
            saver.restore(sess, model_name)
            # -------------- get train_x_emb ------------
            print 'get train_x_emb'
            i = 0
            k = 0
            zero_y = np.zeros((self.batch_size, 2))
            zero_label_emb = np.zeros((self.batch_size, self.model.label_embedding_dim))
            while i < len(self.train_data.pids):
                k += 1
                if k % self.show_batches == 0:
                    print 'batch ' + str(k)
                batch_pid, batch_x, batch_len = self.train_data.get_pid_x(i, i + self.batch_size)
                if self.if_use_seq_len:
                    feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(zero_y),
                                 self.model.seqlen: np.array(batch_len),
                                 self.model.label_embedding_id: zero_label_emb
                                 }
                else:
                    feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(zero_y),
                                 self.model.label_embedding_id: zero_label_emb
                                 }
                x_emb_ = sess.run(x_emb, feed_dict)
                for x_i in range(len(batch_pid)):
                    self.train_x_emb[batch_pid[x_i]] = x_emb_[x_i]
                i += self.batch_size
            print 'dump train_x_emb'
            dump_pickle(self.train_x_emb, trained_model_path+'train_x_emb.pkl')
            # -------------- get test_x_emb -------------
            print 'get test_x_emb'
            i = 0
            k = 0
            while i < len(self.test_data.pids):
                k += 1
                if k % self.show_batches == 0:
                    print 'batch ' + str(k)
                batch_pid, batch_x, batch_len = self.test_data.get_pid_x(i, i + self.batch_size)
                if self.if_use_seq_len:
                    feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(zero_y),
                                 self.model.seqlen: np.array(batch_len),
                                 self.model.label_embedding_id: zero_label_emb
                                 }
                else:
                    feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(zero_y),
                                 self.model.label_embedding_id: zero_label_emb
                                 }
                x_emb_ = sess.run(x_emb, feed_dict)
                for x_i in range(len(batch_pid)):
                    self.test_x_emb[batch_pid[x_i]] = x_emb_[x_i]
                i += self.batch_size
            print 'dump test_x_emb'
            dump_pickle(self.test_x_emb, trained_model_path+'test_x_emb.pkl')

    def get_candidate_label_from_x_emb(self, trained_model_path, k):
        train_pid = self.train_x_emb.keys()
        train_emb = self.train_x_emb.values()
        test_pid = self.test_x_emb.keys()
        test_emb = self.test_x_emb.values()
        print 'begin KNN '
        nbrs = NearestNeighbors(n_neighbors=k).fit(train_emb)
        print 'end KNN'
        _, indices = nbrs.kneighbors(test_emb)
        # get candidate label
        test_unique_candidate_label = {}
        test_all_candidate_label = {}
        for i in range(len(indices)):
            k_nbs = np.array(train_pid)[indices[i]]
            can_l = []
            for pid in k_nbs:
                can_l.append(self.train_data.label_data[pid])
            all_can_l = np.concatenate(can_l)
            unique_can_l = np.unique(all_can_l)
            test_all_candidate_label[test_pid[i]] = all_can_l
            test_unique_candidate_label[test_pid[i]] = unique_can_l
        self.test_unique_candidate_label = test_unique_candidate_label
        self.test_all_candidate_label = test_all_candidate_label
        dump_pickle(self.test_unique_candidate_label, trained_model_path+'test_candidate_label.pkl')

    def predict(self, trained_model_path, output_file_path, k=10, emb_saved=0, can_saved=0):
        print 'generate X-embedding'
        if emb_saved:
            self.train_x_emb = load_pickle(trained_model_path+'train_x_emb.pkl')
            self.test_x_emb = load_pickle(trained_model_path+'test_x_emb.pkl')
        else:
            self.generate_x_embedding(trained_model_path)
        print 'get candidate label from X-embedding'
        if can_saved:
            self.test_unique_candidate_label = load_pickle(trained_model_path+'test_candidate_label.pkl')
        else:
            self.get_candidate_label_from_x_emb(trained_model_path, k)
        test_loader = copy.deepcopy(self.test_data)
        test_loader.candidate_label_data = self.test_unique_candidate_label
        test_loader.reset_data()
        print 'begin predict'
        self.test(trained_model_path, output_file_path, test_loader)




