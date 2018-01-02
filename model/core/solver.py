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
from model.utils.op_utils import precision, precision_for_all


class ModelSolver(object):
    def __init__(self, model, train_data, test_data, **kwargs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.learning_rate = kwargs.pop('learning_rate', 0.000001)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.model_path = kwargs.pop('model_path', './model/')
        # self.save_every = kwargs.pop('save_every', 1)
        # self.log_path = kwargs.pop('log_path', './log/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_path = kwargs.pop('test_path', None)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # if not os.path.exists(self.log_path):
        #     os.makedirs(self.log_path)
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
        y_, loss = self.model.build_model()
        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        tf.get_variable_scope().reuse_variables()
        # set upper limit of used gpu memory
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            if self.pretrained_model is not None:
                print 'Start training with pretrained model...'
                #pretrained_model_path = self.pretrained_model
                pretrained_model_path = self.model_path + self.pretrained_model
                saver.restore(sess, pretrained_model_path)
            for e in range(self.n_epochs):
                curr_loss = 0
                i = 0
                #'''
                #train_loader.pids_copy = train_loader.pids_copy[:5]
                while not train_loader.end_of_data:
                    if i % 20 == 0:
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
                w_text = 'at epoch ' + str(e) + ', train loss is ' + str(curr_loss) + '\n'
                print w_text
                o_file.write(w_text)
                # save model
                save_name = self.model_path + 'model'
                saver.save(sess, save_name, global_step=e+1)
                print 'model-%s saved.' % (e+1)
                #'''

                # ----------------- test ---------------------
                if e % 2 == 0:
                    print '=============== test ================'
                    val_loss = 0
                    pre_pid_label = {}
                    pre_pid_score = {}
                    i = 0
                    #test_loader.pids_copy = test_loader.pids_copy[:5]
                    while not test_loader.end_of_data:
                        if i % 100 == 0:
                            print i
                        batch_pid, batch_label, x, y, seq_l, label_emb = test_loader.next_batch()
                        if len(batch_pid) < self.batch_size:
                            test_loader.reset_data()
                            break
                        feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                     self.model.seqlen: np.array(seq_l), self.model.label_embeddings: label_emb}
                        y_p, l_ = sess.run([y_, loss], feed_dict)
                        val_loss += l_
                        i += 1
                        # get all predictions
                        for j in range(len(batch_pid)):
                            try:
                                pre_pid_label[batch_pid[j]].append(batch_label[j])
                                pre_pid_score[batch_pid[j]].append(y_p[j])
                            except KeyError:
                                pre_pid_label[batch_pid[j]] = [batch_label[j]]
                                pre_pid_score[batch_pid[j]] = [y_p[j]]
                    else:
                        test_loader.reset_data()
                    mean_metric = precision_for_all(test_loader.label_data, pre_pid_label, pre_pid_score)
                    print len(mean_metric)
                    w_text = 'at epoch' + str(e) + ', test loss is ' + str(val_loss) + '\n'
                    print w_text
                    o_file.write(w_text)
                    p1_txt = 'precision@1: ' + str(mean_metric[0]) + '\n'
                    p3_txt = 'precision@3: ' + str(mean_metric[1]) + '\n'
                    p5_txt = 'precision@5: ' + str(mean_metric[2]) + '\n'
                    ndcg1_txt = 'ndcg@1: ' + str(mean_metric[3]) + '\n'
                    ndcg3_txt = 'ndcg@3: ' + str(mean_metric[4]) + '\n'
                    ndcg5_txt = 'ndcg@5: ' + str(mean_metric[5]) + '\n'
                    o_file.write(p1_txt + p3_txt + p5_txt + ndcg1_txt + ndcg3_txt + ndcg5_txt)
                    print p1_txt + p3_txt + p5_txt + ndcg1_txt + ndcg3_txt + ndcg5_txt
            # save model
            save_name = self.model_path + 'model_final'
            saver.save(sess, save_name)
            print 'final model saved.'
            o_file.close()

    def test(self, test_loader):
        # build_model
        y_, loss = self.model.build_model()
        with tf.Session() as sess:
            #imported_meta = tf.train.import_meta_graph('model_final.meta')
            saver = tf.train.Saver()
            print '=============== restore ============='
            saver.restore(sess, tf.train.latest_checkpoint(self.test_path))
            print '=============== test ================'
            test_loss = 0
            pre_pid_label = {}
            pre_pid_score = {}
            i = 0
            test_loader.pids_copy = test_loader.pids_copy[:3]
            while True:
                #if i % 10 == 0:
                print '---------- '+ str(i) + '----------'
                pid, x, seq_l, all_labels, all_y_padding, all_label_emb_padding = test_loader.next_text()
                if test_loader.end_of_data:
                    test_loader.reset_data()
                    break
                prob_i = []
                for ii in range(len(all_y_padding)):
                    if ii % 10 == 0:
                        print ii
                    feed_dict = {self.model.x: np.array(x), self.model.y: np.array(all_y_padding[ii]),
                             self.model.seqlen: np.array(seq_l), self.model.label_embeddings: all_label_emb_padding[ii]}
                    y_p, l_ = sess.run([y_, loss], feed_dict)
                    prob_i.append(y_p)
                test_loss += l_
                i += 1
                # get all predictions
                pre_i = np.concatenate(prob_i)
                pre_i = pre_i[:len(all_labels)]
                try:
                    pre_pid_label[pid].append(all_labels)
                    pre_pid_score[pid].append(pre_i)
                except KeyError:
                    pre_pid_label[pid] = all_labels
                    pre_pid_score[pid] = pre_i
            mean_metric = precision_for_all(test_loader.label_data, pre_pid_label, pre_pid_score)
            print len(mean_metric)
            print 'test loss is ' + str(test_loss)
            print 'precision@1: ' + str(mean_metric[0])
            print 'precision@3: ' + str(mean_metric[1])
            print 'precision@5: ' + str(mean_metric[2])
            print 'ndcg@1: ' + str(mean_metric[3])
            print 'ndcg@3: ' + str(mean_metric[4])
            print 'ndcg@5: ' + str(mean_metric[5])



