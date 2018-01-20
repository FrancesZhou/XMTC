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
from sklearn.neighbors import NearestNeighbors
# from biLSTM.preprocessing.preprocessing import batch_data, get_max_seq_len, construct_train_test_corpus, \
#     generate_labels_from_file, generate_label_pair_from_file
# from biLSTM.utils.io_utils import load_pickle, write_file, load_txt
from model.utils.op_utils import precision_for_label_vector, precision_for_all
from model.utils.io_utils import load_pickle, dump_pickle


class ModelSolver(object):
    def __init__(self, model, train_data, test_data, **kwargs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.train_x_emb = {}
        self.test_x_emb = {}
        self.test_unique_candidate_label = {}
        self.test_all_candidate_label = {}
        self.if_use_seq_len = kwargs.pop('if_use_seq_len', 0)
        self.if_output_all_labels = kwargs.pop('if_output_all_labels', 0)
        self.show_batches = kwargs.pop('show_batches', 20)
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
        train_loader.reset_data()
        test_loader.reset_data()
        # build_model
        _, y_, loss = self.model.build_model()
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
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            if self.pretrained_model is not None:
                print 'Start training with pretrained model...'
                #pretrained_model_path = self.pretrained_model
                pretrained_model_path = self.model_path + self.pretrained_model
                saver.restore(sess, pretrained_model_path)
            for e in range(self.n_epochs):
                print '========== begin epoch ' + str(e) + '==========='
                curr_loss = 0
                if self.if_output_all_labels:
                    k = 0
                    batches = np.arange(math.ceil(len(train_loader.pids)*1.0/self.batch_size), dtype=int)
                    print len(batches)
                    np.random.shuffle(batches)
                    for i in batches:
                        if k % self.show_batches == 0:
                            print 'batch ' + str(k)
                        batch_pid, batch_x, batch_y = train_loader.get_pid_x(i*self.batch_size, (i+1)*self.batch_size)
                        #print np.array(batch_x).shape
                        #print np.array(batch_y).shape
                        feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(batch_y)}
                        try:
                            _, l_ = sess.run([train_op, loss], feed_dict)
                        except:
                            print i
                            print np.array(batch_x).shape
                            print np.array(batch_y).shape
                        curr_loss += l_
                        k += 1
                else:
                    # '''
                    # train_loader.pids_copy = train_loader.pids_copy[:5]
                    i = 0
                    while not train_loader.end_of_data:
                        if i % self.show_batches == 0:
                            print 'batch ' + str(i)
                        batch_pid, _, x, y, seq_l, label_emb = train_loader.next_batch()
                        if len(batch_pid) == 0:
                            continue
                        if len(batch_pid) < self.batch_size:
                            x = np.concatenate(
                                (np.array(x), np.zeros(
                                    (self.batch_size - len(batch_pid), self.model.max_seq_len)
                                )),
                                axis=0)
                            y = np.concatenate((np.array(y), np.zeros((self.batch_size - len(batch_pid), 2))), axis=0)
                            seq_l = np.concatenate((np.array(seq_l), np.zeros((self.batch_size - len(batch_pid)))))
                            label_emb = np.concatenate((np.array(label_emb),
                                                        np.zeros((self.batch_size - len(batch_pid),
                                                                  self.model.label_embedding_dim))), axis=0)
                        if self.if_use_seq_len:
                            feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                         self.model.seqlen: np.array(seq_l),
                                         self.model.label_embeddings: label_emb
                                         }
                        else:
                            feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                         self.model.label_embeddings: label_emb
                                         }
                        _, l_ = sess.run([train_op, loss], feed_dict)
                        curr_loss += l_
                        i += 1
                    else:
                        train_loader.reset_data()
                # ====== output loss ======
                w_text = 'at epoch ' + str(e) + ', train loss is ' + str(curr_loss) + '\n'
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
                    val_loss = 0
                    if self.if_output_all_labels:
                        pre_pid_score = {}
                        tar_pid_label = {}
                        k = 0
                        batches = np.arange(math.ceil(len(train_loader.pids)*1.0 / self.batch_size), dtype=int)
                        #np.random.shuffle(batches)
                        for i in batches:
                            if k % self.show_batches == 0:
                                print 'batch ' + str(i)
                            batch_pid, batch_x, batch_y = train_loader.get_pid_x(int(i * self.batch_size),
                                                                                 int((i + 1) * self.batch_size))
                            feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(batch_y)}
                            y_p, l_ = sess.run([y_, loss], feed_dict)
                            print l_
                            curr_loss += l_
                            k += 1
                            # get all predictions
                            for j in range(len(batch_pid)):
                                tar_pid_label[batch_pid[j]] = np.squeeze(np.nonzero(batch_y[j]))
                                pre_pid_score[batch_pid[j]] = np.argsort(-y_p[j])[:5]
                        mean_metric = precision_for_label_vector(tar_pid_label, pre_pid_score)
                    else:
                        pre_pid_label = {}
                        pre_pid_score = {}
                        i = 0
                        #test_loader.pids_copy = test_loader.pids_copy[:5]
                        while not test_loader.end_of_data:
                            if i % self.show_batches == 0:
                                print i
                            batch_pid, batch_label, x, y, seq_l, label_emb = test_loader.next_batch()
                            if len(batch_pid) == 0:
                                continue
                            if len(batch_pid) < self.batch_size:
                                x = np.concatenate(
                                    (np.array(x), np.zeros(
                                        (self.batch_size - len(batch_pid), self.model.max_seq_len)
                                    )),
                                    axis=0)
                                y = np.concatenate((np.array(y), np.zeros((self.batch_size - len(batch_pid), 2))), axis=0)
                                seq_l = np.concatenate((np.array(seq_l), np.zeros((self.batch_size - len(batch_pid)))))
                                label_emb = np.concatenate((np.array(label_emb),
                                                            np.zeros((self.batch_size - len(batch_pid),
                                                                      self.model.label_embedding_dim))), axis=0)
                            if self.if_use_seq_len:
                                feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                             self.model.seqlen: np.array(seq_l),
                                             self.model.label_embeddings: label_emb
                                             }
                            else:
                                feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                             self.model.label_embeddings: label_emb
                                             }
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

    def test(self, trained_model_path, output_file_path, test_loader=None):
        o_file = open(output_file_path, 'w')
        if not test_loader:
            test_loader = self.test_data
        test_loader.reset_data()
        # restore trained_model
        _, y_, loss = self.model.build_model()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            print 'load trained model...'
            model_name = trained_model_path + 'model_final'
            saver.restore(sess, model_name)
            # -------------- test -------------
            print 'begin testing...'
            pre_pid_label = {}
            pre_pid_score = {}
            i = 0
            # test_loader.pids_copy = test_loader.pids_copy[:5]
            while not test_loader.end_of_data:
                if i % self.show_batches == 0:
                    print i
                batch_pid, batch_label, x, y, seq_l, label_emb = test_loader.next_batch()
                if len(batch_pid) == 0:
                    continue
                if len(batch_pid) < self.batch_size:
                    x = np.concatenate(
                        (np.array(x), np.zeros(
                            (self.batch_size - len(batch_pid), self.model.max_seq_len)
                        )),
                        axis=0)
                    y = np.concatenate((np.array(y), np.zeros((self.batch_size - len(batch_pid), 2))), axis=0)
                    seq_l = np.concatenate((np.array(seq_l), np.zeros((self.batch_size - len(batch_pid)))))
                    label_emb = np.concatenate((np.array(label_emb),
                                                np.zeros((self.batch_size - len(batch_pid),
                                                          self.model.label_embedding_dim))), axis=0)
                if self.if_use_seq_len:
                    feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                 self.model.seqlen: np.array(seq_l),
                                 self.model.label_embeddings: label_emb
                                 }
                else:
                    feed_dict = {self.model.x: np.array(x), self.model.y: np.array(y),
                                 self.model.label_embeddings: label_emb
                                }
                y_p = sess.run(y_, feed_dict)
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
            #w_text = 'test loss is ' + str(np.mean(test_loss)) + '\n'
            #print w_text
            #o_file.write(w_text)
            p1_txt = 'precision@1: ' + str(mean_metric[0]) + '\n'
            p3_txt = 'precision@3: ' + str(mean_metric[1]) + '\n'
            p5_txt = 'precision@5: ' + str(mean_metric[2]) + '\n'
            ndcg1_txt = 'ndcg@1: ' + str(mean_metric[3]) + '\n'
            ndcg3_txt = 'ndcg@3: ' + str(mean_metric[4]) + '\n'
            ndcg5_txt = 'ndcg@5: ' + str(mean_metric[5]) + '\n'
            o_file.write(p1_txt + p3_txt + p5_txt + ndcg1_txt + ndcg3_txt + ndcg5_txt)
            print p1_txt + p3_txt + p5_txt + ndcg1_txt + ndcg3_txt + ndcg5_txt
            o_file.close()


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
                                 self.model.label_embeddings: zero_label_emb
                                 }
                else:
                    feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(zero_y),
                                 self.model.label_embeddings: zero_label_emb
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
                                 self.model.label_embeddings: zero_label_emb
                                 }
                else:
                    feed_dict = {self.model.x: np.array(batch_x), self.model.y: np.array(zero_y),
                                 self.model.label_embeddings: zero_label_emb
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




