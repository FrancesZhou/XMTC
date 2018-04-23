'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
# import re
# import copy
from ..utils.op_utils import *
from .preprocessing import generate_embedding_from_vocabID, generate_label_vector_of_fixed_length, get_wordID_from_vocab

# for CNN_comp, output 20 rank score of candidate labels
class DataLoader():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data, num_candidate,
                 label_dict,
                 max_seq_len=3000,
                 if_use_all_true_label=0):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.given_candidate_label_data = candidate_label_data
        self.candidate_label_data = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_y = {}
        self.doc_length = {}
        self.num_candidate = num_candidate
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.all_labels = label_dict.keys()
        self.max_seq_len = max_seq_len
        self.if_use_all_true_label = if_use_all_true_label
        self.train_pids = []
        self.val_pids = []
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        if self.if_use_all_true_label:
            for pid in self.pids:
                wordID_seq = self.doc_wordID_data[pid]
                seq_len = min(len(wordID_seq), self.max_seq_len)
                if seq_len:
                    self.doc_length[pid] = seq_len
                    padding_len = self.max_seq_len - seq_len
                    x = np.array(wordID_seq, dtype=int) + 1
                    if padding_len > 0:
                        x = np.concatenate((x, np.zeros(padding_len)))
                    self.doc_wordID_data[pid] = x[:self.max_seq_len]
                    # labels
                    true_labels = self.label_data[pid]
                    num_true = len(true_labels)
                    num_candidate = max(self.num_candidate, num_true)
                    self.candidate_label_data[pid] = np.zeros(num_candidate)
                    self.candidate_label_embedding_id[pid] = np.zeros(num_candidate)
                    self.candidate_label_prop[pid] = np.zeros([num_candidate, 2])
                    self.candidate_label_data[pid][:num_true] = true_labels
                    self.candidate_label_embedding_id[pid][:num_true] = [self.label_dict[e] for e in true_labels]
                    self.candidate_label_prop[pid][:num_true] = np.tile([0, 1], [num_true, 1])
                    if num_true < num_candidate:
                        neg_labels = [e for e in self.given_candidate_label_data[pid] if e not in true_labels]
                        for j in range(num_true, self.num_candidate):
                            l_ = neg_labels[j - num_true]
                            self.candidate_label_data[pid][j] = l_
                            self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                            self.candidate_label_prop[pid][j] = [1, 0]
                else:
                    del self.doc_wordID_data[pid]
                    del self.label_data[pid]
                    del self.given_candidate_label_data[pid]
        else:
            for pid in self.pids:
                wordID_seq = self.doc_wordID_data[pid]
                seq_len = min(len(wordID_seq), self.max_seq_len)
                if seq_len:
                    self.doc_length[pid] = seq_len
                    padding_len = self.max_seq_len - seq_len
                    x = np.array(wordID_seq, dtype=int) + 1
                    if padding_len > 0:
                        x = np.concatenate((x, np.zeros(padding_len)))
                    self.doc_wordID_data[pid] = x[:self.max_seq_len]
                    # labels
                    self.candidate_label_data[pid] = self.given_candidate_label_data[pid][:self.num_candidate]
                    self.candidate_label_embedding_id[pid] = np.zeros(self.num_candidate)
                    self.candidate_label_y[pid] = np.zeros(self.num_candidate)
                    for j in range(self.num_candidate):
                        l_ = self.candidate_label_data[pid][j]
                        self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                        if l_ in self.label_data[pid]:
                            self.candidate_label_y[pid][j] = 1
                        else:
                            self.candidate_label_y[pid][j] = 0
                else:
                    del self.doc_wordID_data[pid]
                    del self.label_data[pid]
                    del self.given_candidate_label_data[pid]
        self.pids = self.label_data.keys()
        print 'after removing zero-length data'
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        self.reset_data()

    def get_pid_x(self, i, j):
        batch_pid = []
        batch_x = []
        batch_length = []
        end = min(j, len(self.pids))
        for pid in self.pids[i:end]:
            batch_pid.append(pid)
            seq_len = min(self.doc_length[pid], self.max_seq_len)
            padding_len = self.max_seq_len - seq_len
            x = np.array(self.doc_wordID_data[pid]) + 1
            if padding_len:
                x = np.concatenate((x, np.zeros(padding_len, dtype=int)))
            batch_length.append(seq_len)
            batch_x.append(x)
        if end < j:
            batch_length = np.concatenate((batch_length, np.zeros(j-end, dtype=int)), axis=0)
            batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
        return batch_pid, batch_x, batch_length

    def next_batch(self, pids, start_pid, end_pid):
        end = min(len(pids), end_pid)
        #pid_num = end - start_pid
        # batch_pid = np.zeros([pid_num, self.num_candidate])
        # batch_length = np.zeros([pid_num, self.num_candidate])
        # batch_x = np.zeros([pid_num, self.num_candidate, self.max_seq_len])
        # batch_label = np.zeros([pid_num, self.num_candidate])
        # batch_label_embedding_id = np.zeros([pid_num, self.num_candidate])
        # batch_y = np.zeros([pid_num, self.num_candidate, 2])
        # i = 0
        batch_pid = []
        batch_length = []
        batch_x = []
        batch_label = []
        batch_label_embedding_id = []
        batch_y = []
        for pid in pids[start_pid:end]:
            #num = len(self.candidate_label_data[pid])
            batch_pid.append(pid)
            batch_length.append(self.doc_length[pid])
            batch_x.append(self.doc_wordID_data[pid])
            batch_label.append(self.candidate_label_data[pid])
            batch_label_embedding_id.append(self.candidate_label_embedding_id[pid])
            batch_y.append(self.candidate_label_y[pid])
            # batch_pid[i] = [pid]*self.num_candidate
            # batch_length[i] = [self.doc_length[pid]]*self.num_candidate
            # batch_x[i] = np.tile(self.doc_wordID_data[pid], [self.num_candidate, 1])
            # batch_label[i] = self.candidate_label_data[pid]
            # batch_label_embedding_id[i] = self.candidate_label_embedding_id[pid]
            # batch_y[i] = self.candidate_label_prop[pid]
            # i += 1
        return batch_pid, batch_label, batch_x, batch_y, batch_length, batch_label_embedding_id

    def reset_data(self):
        self.train_pids, self.val_pids = train_test_split(self.pids, test_size=0.1)

class DataLoader2():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data, num_candidate,
                 label_dict,
                 max_seq_len=3000,
                 if_use_all_true_label=0):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.given_candidate_label_data = candidate_label_data
        self.candidate_label_data = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_prop = {}
        self.doc_length = {}
        self.num_candidate = num_candidate
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.all_labels = label_dict.keys()
        self.max_seq_len = max_seq_len
        self.if_use_all_true_label = if_use_all_true_label
        self.train_pids = []
        self.val_pids = []
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        if self.if_use_all_true_label:
            for pid in self.pids:
                wordID_seq = self.doc_wordID_data[pid]
                seq_len = min(len(wordID_seq), self.max_seq_len)
                if seq_len:
                    self.doc_length[pid] = seq_len
                    padding_len = self.max_seq_len - seq_len
                    x = np.array(wordID_seq, dtype=int) + 1
                    if padding_len > 0:
                        x = np.concatenate((x, np.zeros(padding_len)))
                    self.doc_wordID_data[pid] = x[:self.max_seq_len]
                    # labels
                    true_labels = self.label_data[pid]
                    num_true = len(true_labels)
                    num_candidate = max(self.num_candidate, num_true)
                    self.candidate_label_data[pid] = np.zeros(num_candidate)
                    self.candidate_label_embedding_id[pid] = np.zeros(num_candidate)
                    self.candidate_label_prop[pid] = np.zeros([num_candidate, 2])
                    self.candidate_label_data[pid][:num_true] = true_labels
                    self.candidate_label_embedding_id[pid][:num_true] = [self.label_dict[e] for e in true_labels]
                    self.candidate_label_prop[pid][:num_true] = np.tile([0, 1], [num_true, 1])
                    if num_true < num_candidate:
                        neg_labels = [e for e in self.given_candidate_label_data[pid] if e not in true_labels]
                        for j in range(num_true, self.num_candidate):
                            l_ = neg_labels[j - num_true]
                            self.candidate_label_data[pid][j] = l_
                            self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                            self.candidate_label_prop[pid][j] = [1, 0]
                else:
                    del self.doc_wordID_data[pid]
                    del self.label_data[pid]
                    del self.given_candidate_label_data[pid]
        else:
            for pid in self.pids:
                wordID_seq = self.doc_wordID_data[pid]
                seq_len = min(len(wordID_seq), self.max_seq_len)
                if seq_len:
                    self.doc_length[pid] = seq_len
                    padding_len = self.max_seq_len - seq_len
                    x = np.array(wordID_seq, dtype=int) + 1
                    if padding_len > 0:
                        x = np.concatenate((x, np.zeros(padding_len)))
                    self.doc_wordID_data[pid] = x[:self.max_seq_len]
                    # labels
                    self.candidate_label_data[pid] = self.given_candidate_label_data[pid][:self.num_candidate]
                    self.candidate_label_embedding_id[pid] = np.zeros(self.num_candidate)
                    self.candidate_label_prop[pid] = np.zeros([self.num_candidate, 2])
                    for j in range(self.num_candidate):
                        l_ = self.candidate_label_data[pid][j]
                        self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                        if l_ in self.label_data[pid]:
                            self.candidate_label_prop[pid][j] = [0, 1]
                        else:
                            self.candidate_label_prop[pid][j] = [1, 0]
                else:
                    del self.doc_wordID_data[pid]
                    del self.label_data[pid]
                    del self.given_candidate_label_data[pid]
        self.pids = self.label_data.keys()
        print 'after removing zero-length data'
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        self.reset_data()

    def get_pid_x(self, i, j):
        batch_pid = []
        batch_x = []
        batch_length = []
        end = min(j, len(self.pids))
        for pid in self.pids[i:end]:
            batch_pid.append(pid)
            seq_len = min(self.doc_length[pid], self.max_seq_len)
            padding_len = self.max_seq_len - seq_len
            x = np.array(self.doc_wordID_data[pid]) + 1
            if padding_len:
                x = np.concatenate((x, np.zeros(padding_len, dtype=int)))
            batch_length.append(seq_len)
            batch_x.append(x)
        if end < j:
            batch_length = np.concatenate((batch_length, np.zeros(j-end, dtype=int)), axis=0)
            batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
        return batch_pid, batch_x, batch_length

    def next_batch(self, pids, start_pid, end_pid):
        end = min(len(pids), end_pid)
        pid_num = end - start_pid
        # batch_pid = np.zeros([pid_num, self.num_candidate])
        # batch_length = np.zeros([pid_num, self.num_candidate])
        # batch_x = np.zeros([pid_num, self.num_candidate, self.max_seq_len])
        # batch_label = np.zeros([pid_num, self.num_candidate])
        # batch_label_embedding_id = np.zeros([pid_num, self.num_candidate])
        # batch_y = np.zeros([pid_num, self.num_candidate, 2])
        # i = 0
        batch_pid = []
        batch_length = []
        batch_x = []
        batch_label = []
        batch_label_embedding_id = []
        batch_y = []
        for pid in pids[start_pid:end]:
            num = len(self.candidate_label_data[pid])
            batch_pid.append([pid]*num)
            batch_length.append([self.doc_length[pid]]*num)
            batch_x.append(np.tile(self.doc_wordID_data[pid], [num, 1]))
            batch_label.append(self.candidate_label_data[pid])
            batch_label_embedding_id.append(self.candidate_label_embedding_id[pid])
            batch_y.append(self.candidate_label_prop[pid])
            # batch_pid[i] = [pid]*self.num_candidate
            # batch_length[i] = [self.doc_length[pid]]*self.num_candidate
            # batch_x[i] = np.tile(self.doc_wordID_data[pid], [self.num_candidate, 1])
            # batch_label[i] = self.candidate_label_data[pid]
            # batch_label_embedding_id[i] = self.candidate_label_embedding_id[pid]
            # batch_y[i] = self.candidate_label_prop[pid]
            # i += 1
        if pid_num > 1:
            batch_pid = np.concatenate(batch_pid, axis=0)
            batch_length = np.concatenate(batch_length, axis=0)
            batch_x = np.concatenate(batch_x, axis=0)
            batch_label = np.concatenate(batch_label, axis=0)
            batch_label_embedding_id = np.concatenate(batch_label_embedding_id, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
        else:
            batch_pid = np.squeeze(batch_pid)
            batch_length = np.squeeze(batch_length)
            batch_x = np.squeeze(batch_x)
            batch_label = np.squeeze(batch_label)
            batch_label_embedding_id = np.squeeze(batch_label_embedding_id)
            batch_y = np.squeeze(batch_y)
        return batch_pid, batch_label, batch_x, batch_y, batch_length, batch_label_embedding_id

    def reset_data(self):
        self.train_pids, self.val_pids = train_test_split(self.pids, test_size=0.1)

# DataLoader3 is for loading candidate label subset from SLEEC or fastxml
class DataLoader3():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data,
                 label_dict,
                 batch_size,
                 given_seq_len=False, max_seq_len=5000,
                 if_use_all_true_label=0):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.candidate_label_data = candidate_label_data
        self.pids = self.label_data.keys()
        self.pid_label = []
        self.batch_num = 0
        self.label_dict = label_dict
        self.all_labels = label_dict.keys()
        self.batch_size = batch_size
        self.given_seq_len = given_seq_len
        self.max_seq_len = max_seq_len
        self.if_use_all_true_label = if_use_all_true_label
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        self.doc_length = {}
        all_length = []
        for pid in self.pids:
            seq_len = len(self.doc_wordID_data[pid])
            if seq_len:
                all_length.append(seq_len)
                self.doc_length[pid] = seq_len
            else:
                del self.doc_wordID_data[pid]
                del self.label_data[pid]
                del self.candidate_label_data[pid]
        self.pids = self.label_data.keys()
        print 'after removing zero-length data'
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # assign max_seq_len if not given_seq_len
        if not self.given_seq_len:
            self.max_seq_len = min(max(all_length), self.max_seq_len)
        # if_use_all_true_label
        if self.if_use_all_true_label:
            for pid, label in self.label_data.items():
                candidate_label = self.candidate_label_data[pid]
                candidate_label = list(set(candidate_label) & set(self.all_labels))
                self.candidate_label_data[pid] = np.unique(np.concatenate((candidate_label, label))).tolist()
        # generate self.pid_label
        for pid, label in self.candidate_label_data.items():
            stack_pid = [pid]*len(label)
            self.pid_label.append(np.concatenate(
                (np.expand_dims(stack_pid, -1), np.expand_dims(label, -1)), axis=-1))
        self.pid_label = np.concatenate(self.pid_label, axis=0)
        self.batch_num = math.ceil(len(self.pid_label) / float(self.batch_size))
        print 'pid_label shape: ' + str(self.pid_label.shape)
        self.reset_data()

    def get_pid_x(self, i, j):
        batch_pid = []
        batch_x = []
        batch_length = []
        end = min(j, len(self.pids))
        for pid in self.pids[i:end]:
            batch_pid.append(pid)
            seq_len = min(self.doc_length[pid], self.max_seq_len)
            padding_len = self.max_seq_len - seq_len
            x = np.array(self.doc_wordID_data[pid]) + 1
            if padding_len:
                x = np.concatenate((x, np.zeros(padding_len, dtype=int)))
            batch_length.append(seq_len)
            batch_x.append(x)
        if end < j:
            batch_length = np.concatenate((batch_length, np.zeros(j-end, dtype=int)), axis=0)
            batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
        return batch_pid, batch_x, batch_length

    def next_batch(self):
        batch_pid = []
        batch_label = []
        batch_x = []
        batch_y = []
        batch_length = []
        batch_label_embedding_id = []
        if self.batch_id == self.batch_num-1:
            index = np.arange(self.batch_id*self.batch_size, len(self.pid_label))
            self.batch_id = 0
            self.end_of_data = True
        else:
            index = np.arange(self.batch_id*self.batch_size, (self.batch_id+1)*self.batch_size)
            self.batch_id += 1
        for i in index:
            pid, label = self.pid_label[i]
            pid = int(pid)
            seq_len = min(self.doc_length[pid], self.max_seq_len)
            padding_len = self.max_seq_len - seq_len
            x = np.array(self.doc_wordID_data[pid], dtype=int) + 1
            x = x.tolist()
            if padding_len:
                x = x + [0]*padding_len
            x = x[:self.max_seq_len]
            batch_pid.append(pid)
            batch_label.append(label)
            batch_x.append(x)
            if label in self.label_data[pid]:
                batch_y.append([0, 1])
            else:
                batch_y.append([1, 0])
            batch_length.append(seq_len)
            batch_label_embedding_id.append(int(self.label_dict[label]))
        return batch_pid, batch_label, batch_x, batch_y, batch_length, batch_label_embedding_id

    def reset_data(self):
        np.random.shuffle(self.pid_label)
        self.batch_id = 0
        self.end_of_data = False

# DataLoader5 is for XML-CNN to output all labels
class DataLoader_all():
    def __init__(self, doc_wordID_data, label_data,
                 num_labels, label_prop_dict,
                 batch_size,
                 max_seq_len=5000):
        self.doc_wordID_data = doc_wordID_data
        self.x_feature_indices = {}
        self.x_feature_values = {}
        self.label_data = label_data
        self.label_prop = label_prop_dict
        self.pids = []
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.doc_length = {}
        self.initialize_dataloader()
        self.reset_data()

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        print 'max sequence length: ' + str(self.max_seq_len)
        # label
        zero_prop_label = set(range(self.num_labels)) - set(self.label_prop.keys())
        for zero_l in zero_prop_label:
            self.label_prop[zero_l] = 0
        #
        self.pids = np.asarray(self.label_data.keys())
        for pid in self.pids:
            temp = sorted(self.doc_wordID_data[pid].items(), key=lambda e: e[1], reverse=True)
            temp2 = sorted(temp[:self.max_seq_len], key=lambda e: e[0], reverse=False)
            feature_id, feature_v = zip(*temp2)
            seq_len = min(len(feature_id), self.max_seq_len)
            feature_indices = np.array(list(feature_id) + (self.max_seq_len - seq_len) * [0])
            feature_indices[:seq_len] = feature_indices[:seq_len] + 1
            self.x_feature_indices[pid] = feature_indices
            self.x_feature_values[pid] = np.array(list(feature_v) + (self.max_seq_len - seq_len) * [0])
            self.doc_length[pid] = seq_len

    def get_pid_x(self, pool, i, j):
        batch_y = []
        end = min(j, len(pool))
        batch_pid = pool[i:end]
        batch_seq_len = [self.doc_length[p] for p in batch_pid]
        x_feature_id = [self.x_feature_indices[p] for p in batch_pid]
        x_feature_v = [self.x_feature_values[p] for p in batch_pid]
        for pid in batch_pid:
            y = np.zeros(self.num_labels)
            for l in self.label_data[pid]:
                y[l] = 1
            batch_y.append(y)
        # if end < j:
        #     batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
        #     batch_y = np.concatenate((batch_y, np.zeros((j-end, len(self.all_labels)))), axis=0)
        return batch_pid, x_feature_id, x_feature_v, batch_seq_len, batch_y

    def reset_data(self):
        np.random.shuffle(self.pids)
        self.train_pids, self.val_pids = train_test_split(self.pids, test_size=0.1)

# for propensity-loss train dataloader
class TrainDataLoader():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data,
                 label_dict, label_prop,
                 num_pos, num_neg,
                 max_seq_len=3000):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.candidate_label_data = candidate_label_data
        self.doc_length = {}
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.label_prop = label_prop
        self.all_labels = label_dict.keys()
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.label_pos_pid = {}
        self.label_neg_pid = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_y = {}
        self.max_seq_len = max_seq_len
        self.train_pids = []
        self.val_pids = []
        self.pid_label_y = []
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        for pid in self.pids:
            wordID_seq = self.doc_wordID_data[pid]
            seq_len = min(len(wordID_seq), self.max_seq_len)
            self.doc_length[pid] = seq_len
            padding_len = self.max_seq_len - seq_len
            x = np.array(wordID_seq, dtype=int) + 1
            if padding_len > 0:
                x = np.concatenate((x, np.zeros(padding_len)))
            self.doc_wordID_data[pid] = x[:self.max_seq_len]
            # labels
            candidate_label = self.candidate_label_data[pid]
            pos_labels = self.label_data[pid]
            neg_labels = list(set(candidate_label) - set(pos_labels))
            for label in pos_labels:
                try:
                    self.label_pos_pid[label].append(pid)
                except KeyError:
                    self.label_pos_pid[label] = [pid]
            for label in neg_labels:
                try:
                    self.label_neg_pid[label].append(pid)
                except KeyError:
                    self.label_neg_pid[label] = [pid]
            # candidate label for validation
            self.candidate_label_embedding_id[pid] = np.zeros(len(candidate_label))
            self.candidate_label_y[pid] = np.zeros(len(candidate_label))
            for j in range(len(candidate_label)):
                l_ = candidate_label[j]
                self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                if l_ in pos_labels:
                    self.candidate_label_y[pid][j] = 1
        self.reset_data()

    def get_pid_x(self, pid):
        candidate_labels = self.candidate_label_data[pid]
        batch_x = [self.doc_wordID_data[pid]] * len(candidate_labels)
        batch_y = self.candidate_label_y[pid]
        batch_length = [self.doc_length[pid]] * len(candidate_labels)
        batch_label_embedding_id = self.candidate_label_embedding_id[pid]
        batch_label_prop = [self.label_prop[e] for e in candidate_labels]
        return pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop

    def next_batch(self, length, start, end):
        end = min(length, end)
        pid_label_y = self.pid_label_y[start:end]
        batch_pid = pid_label_y[:, 0]
        batch_label = pid_label_y[:, 1]
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[e] for e in batch_label]
        batch_x = [self.doc_wordID_data[p] for p in batch_pid]
        batch_y = pid_label_y[:, 2]
        batch_label_prop = [self.label_prop[e] for e in batch_label]
        return batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop

    def get_fixed_length_pos_samples(self, items, num):
        length = len(items)
        if length < num:
            sample = items
            pad_sample = np.random.choice(items, num-length)
            sample = sample + pad_sample.tolist()
        elif length > num:
            sample = random.sample(items, num)
        else:
            sample = items
        return sample

    def get_fixed_length_neg_samples(self, label, num):
        try:
            items = self.label_neg_pid[label]
            length = len(items)
            if length < num:
                sample = items
                pad_sample = np.random.choice(items, num-length)
                sample = sample + pad_sample.tolist()
            elif length > num:
                sample = random.sample(items, num)
            else:
                sample = items
        except KeyError:
            neg_set = list(set(self.pids) - set(self.label_pos_pid[label]))
            sample = np.random.choice(neg_set, num).tolist()
        return sample

    def reset_data(self):
        self.pid_label_y = []
        for label in self.all_labels:
            pos_pid = self.get_fixed_length_pos_samples(self.label_pos_pid[label], self.num_pos)
            neg_pid = self.get_fixed_length_neg_samples(label, self.num_neg)
            pids = pos_pid + neg_pid
            stack_label = [label] * len(pids)
            y = [1] * self.num_pos + [0] * self.num_neg
            self.pid_label_y.append(np.transpose(np.array([pids, stack_label, y])))
        self.pid_label_y = np.concatenate(self.pid_label_y, axis=0)
        np.random.shuffle(self.pid_label_y)
        # validate
        _, self.val_pids = train_test_split(self.pids, test_size=0.1)

# for propensity-loss test dataloader
class TestDataLoader():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data,
                 label_dict, label_prop,
                 max_seq_len=3000):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.candidate_label_data = candidate_label_data
        self.doc_length = {}
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.label_prop = label_prop
        self.all_labels = label_dict.keys()
        self.label_pos_pid = {}
        self.label_neg_pid = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_y = {}
        self.max_seq_len = max_seq_len
        self.pid_label_y = []
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        for pid in self.pids:
            wordID_seq = self.doc_wordID_data[pid]
            seq_len = min(len(wordID_seq), self.max_seq_len)
            self.doc_length[pid] = seq_len
            padding_len = self.max_seq_len - seq_len
            x = np.array(wordID_seq, dtype=int) + 1
            if padding_len > 0:
                x = np.concatenate((x, np.zeros(padding_len)))
            self.doc_wordID_data[pid] = x[:self.max_seq_len]
            # labels
            candidate_label = self.candidate_label_data[pid]
            true_labels = self.label_data[pid]
            # candidate label for validation
            self.candidate_label_embedding_id[pid] = np.zeros(len(candidate_label))
            self.candidate_label_y[pid] = np.zeros(len(candidate_label))
            for j in range(len(candidate_label)):
                l_ = candidate_label[j]
                self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                if l_ in true_labels:
                    self.candidate_label_y[pid][j] = 1

    def get_pid_x(self, pid):
        candidate_labels = self.candidate_label_data[pid]
        batch_x = [self.doc_wordID_data[pid]] * len(candidate_labels)
        batch_y = self.candidate_label_y[pid]
        batch_length = [self.doc_length[pid]] * len(candidate_labels)
        batch_label_embedding_id = self.candidate_label_embedding_id[pid]
        batch_label_prop = [self.label_prop[e] for e in candidate_labels]
        return pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop

# for propensity-loss and rerank train dataloader
class TrainDataLoader2():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data,
                 label_dict, label_prop,
                 num_pos, num_neg,
                 max_seq_len=3000,
                 if_doc_is_dict=False):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.candidate_label_data = candidate_label_data
        self.doc_length = {}
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.label_prop = label_prop
        self.all_labels = label_dict.keys()
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.label_pos_pid = {}
        self.label_neg_pid = {}
        self.candidate_label = {}
        self.candidate_count_score = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_y = {}
        self.max_seq_len = max_seq_len
        self.if_doc_is_dict = if_doc_is_dict
        # self.train_pids = []
        # self.val_pids = []
        self.pid_label_y = []
        self.initialize_dataloader()
        self.reset_data()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        for pid in self.pids:
            if self.if_doc_is_dict:
                temp = sorted(self.doc_wordID_data[pid].items(), key=lambda e: e[1], reverse=True)
                wordID_seq = dict(temp[:self.max_seq_len]).keys()
            else:
                wordID_seq = self.doc_wordID_data[pid]
            seq_len = min(len(wordID_seq), self.max_seq_len)
            self.doc_length[pid] = seq_len
            padding_len = self.max_seq_len - seq_len
            x = np.array(wordID_seq, dtype=int) + 1
            if padding_len > 0:
                x = np.concatenate((x, np.zeros(padding_len)))
            self.doc_wordID_data[pid] = x[:self.max_seq_len]
            # labels
            # self.candidate_label_data[pid] = dict(
            #     sorted(self.candidate_label_data[pid].items(), key=lambda item: item[1], reverse=True)[:30]
            # )
            candidate_label, count_score = zip(*(self.candidate_label_data[pid].iteritems()))
            self.candidate_label[pid] = candidate_label
            self.candidate_count_score[pid] = count_score
            pos_labels = self.label_data[pid]
            neg_labels = list(set(candidate_label) - set(pos_labels))
            for label in pos_labels:
                try:
                    self.label_pos_pid[label].append(pid)
                except KeyError:
                    self.label_pos_pid[label] = [pid]
            for label in neg_labels:
                try:
                    self.label_neg_pid[label].append(pid)
                except KeyError:
                    self.label_neg_pid[label] = [pid]
            # candidate label for validation
            self.candidate_label_embedding_id[pid] = np.zeros(len(candidate_label))
            self.candidate_label_y[pid] = np.zeros(len(candidate_label))
            for j in range(len(candidate_label)):
                l_ = candidate_label[j]
                self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                if l_ in pos_labels:
                    self.candidate_label_y[pid][j] = 1

    def get_pid_x(self, length, start, end):
        #candidate_labels, batch_count_score = zip(*(self.candidate_label_data[pid].iteritems()))
        #candidate_labels, batch_count_score = zip(*(self.candidate_label_data[pid][:30]))
        batch_pid = []
        batch_x = []
        batch_y = []
        batch_length = []
        batch_label_embedding_id = []
        batch_label_prop = []
        batch_count_score = []
        end2 = min(length, end)
        for i in xrange(start, end2):
            pid = self.val_pids[i]
            candidate_labels = self.candidate_label[pid]
            num = len(candidate_labels)
            batch_pid.append([pid]*num)
            batch_x.append([self.doc_wordID_data[pid]] * num)
            batch_y.append(self.candidate_label_y[pid])
            batch_length.append([self.doc_length[pid]] * num)
            batch_label_embedding_id.append(self.candidate_label_embedding_id[pid])
            batch_label_prop.append([self.label_prop[e] for e in candidate_labels])
            batch_count_score.append(self.candidate_count_score[pid])
        # candidate_labels = self.candidate_label[pid]
        # batch_count_score = self.candidate_count_score[pid]
        # batch_x = [self.doc_wordID_data[pid]] * len(candidate_labels)
        # batch_y = self.candidate_label_y[pid]
        # batch_length = [self.doc_length[pid]] * len(candidate_labels)
        # batch_label_embedding_id = self.candidate_label_embedding_id[pid]
        # batch_label_prop = [self.label_prop[e] for e in candidate_labels]
        if end2 < end:
            batch_pid = np.concatenate(batch_pid, axis=0)
            padding_num = end - end2
            batch_x = np.concatenate((np.concatenate(batch_x, axis=0),
                                      np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.concatenate(batch_y, axis=0),
                                      np.zeros(padding_num)), axis=0)
            batch_length = np.concatenate((np.concatenate(batch_length, axis=0),
                                           np.zeros(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.concatenate(batch_label_embedding_id, axis=0),
                                                       np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.concatenate(batch_label_prop, axis=0),
                                               np.zeros(padding_num)), axis=0)
            batch_count_score = np.concatenate((np.concatenate(batch_count_score, axis=0),
                                                np.zeros(padding_num)), axis=0)
        else:
            batch_pid = np.concatenate(batch_pid, axis=0)
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            batch_length = np.concatenate(batch_length, axis=0)
            batch_label_embedding_id = np.concatenate(batch_label_embedding_id, axis=0)
            batch_label_prop = np.concatenate(batch_label_prop, axis=0)
            batch_count_score = np.concatenate(batch_count_score, axis=0)
        return batch_pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop, batch_count_score

    def set_val_batch(self):
        self.val_pid_label_y = []
        for pid in self.val_pids:
            candidate_labels = self.candidate_label[pid]
            pids = [pid] * len(candidate_labels)
            y = self.candidate_label_y[pid]
            score = self.candidate_count_score[pid]
            self.val_pid_label_y.append(np.transpose(np.array([pids, candidate_labels, y, score])))
        self.val_pid_label_y = np.concatenate(self.val_pid_label_y, axis=0)

    def get_val_batch(self, length, start, end):
        end2 = min(length, end)
        pid_label_y = self.val_pid_label_y[start:end2]
        batch_pid = pid_label_y[:, 0]
        batch_label = pid_label_y[:, 1]
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[e] for e in batch_label]
        batch_x = [self.doc_wordID_data[p] for p in batch_pid]
        batch_y = pid_label_y[:, 2]
        batch_label_prop = [self.label_prop[e] for e in batch_label]
        batch_count_score = pid_label_y[:, 3]
        if end2 < end:
            padding_num = end - end2
            batch_length = np.concatenate((np.array(batch_length),
                                           np.ones(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.array(batch_label_embedding_id),
                                                       np.zeros(padding_num)), axis=0)
            batch_x = np.concatenate((np.array(batch_x),
                                      np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.array(batch_y),
                                      np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.array(batch_label_prop),
                                               np.zeros(padding_num)), axis=0)
            batch_count_score = np.concatenate((np.array(batch_count_score),
                                                np.zeros(padding_num)), axis=0)
        return batch_pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop, batch_count_score

    def next_batch(self, length, start, end):
        end2 = min(length, end)
        pid_label_y = self.pid_label_y[start:end2]
        batch_pid = pid_label_y[:, 0]
        batch_label = pid_label_y[:, 1]
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[e] for e in batch_label]
        batch_x = [self.doc_wordID_data[p] for p in batch_pid]
        batch_y = pid_label_y[:, 2]
        batch_label_prop = [self.label_prop[e] for e in batch_label]
        if end2 < end:
            padding_num = end - end2
            batch_length = np.concatenate((np.array(batch_length),
                                           np.ones(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.array(batch_label_embedding_id),
                                                       np.zeros(padding_num)), axis=0)
            batch_x = np.concatenate((np.array(batch_x),
                                      np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.array(batch_y),
                                      np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.array(batch_label_prop),
                                               np.zeros(padding_num)), axis=0)
        return batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop

    def get_fixed_length_pos_samples(self, items, num):
        length = len(items)
        if length < num:
            sample = items
            pad_sample = np.random.choice(items, num-length)
            sample = sample + pad_sample.tolist()
        elif length > num:
            sample = random.sample(items, num)
        else:
            sample = items
        return sample

    def get_fixed_length_neg_samples(self, label, num):
        try:
            items = self.label_neg_pid[label]
            length = len(items)
            if length < num:
                sample = items
                pad_sample = np.random.choice(items, num-length)
                sample = sample + pad_sample.tolist()
            elif length > num:
                sample = random.sample(items, num)
            else:
                sample = items
        except KeyError:
            neg_set = list(set(self.pids) - set(self.label_pos_pid[label]))
            sample = np.random.choice(neg_set, num).tolist()
        return sample

    def reset_data(self):
        self.pid_label_y = []
        for label in self.all_labels:
            pos_pid = self.get_fixed_length_pos_samples(self.label_pos_pid[label], self.num_pos)
            neg_pid = self.get_fixed_length_neg_samples(label, self.num_neg)
            pids = pos_pid + neg_pid
            stack_label = [label] * len(pids)
            y = [1] * self.num_pos + [0] * self.num_neg
            self.pid_label_y.append(np.transpose(np.array([pids, stack_label, y])))
        self.pid_label_y = np.concatenate(self.pid_label_y, axis=0)
        np.random.shuffle(self.pid_label_y)
        _, self.val_pids = train_test_split(self.pids, test_size=0.1)
        self.set_val_batch()

# for propensity-loss and rerank test dataloader
class TestDataLoader2():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data,
                 label_dict, label_prop,
                 max_seq_len=3000,
                 if_cal_metrics=1,
                 if_doc_is_dict=False):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.candidate_label_data = candidate_label_data
        self.doc_length = {}
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.label_prop = label_prop
        self.all_labels = label_dict.keys()
        self.label_pos_pid = {}
        self.label_neg_pid = {}
        self.candidate_label = {}
        self.candidate_count_score = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_y = {}
        self.max_seq_len = max_seq_len
        self.if_doc_is_dict = if_doc_is_dict
        self.pid_label_y = []
        self.initialize_dataloader()
        if if_cal_metrics:
            self.get_all_metrics_for_baseline_result()
        self.set_all_batch()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        for pid in self.pids:
            if self.if_doc_is_dict:
                wordID_seq = self.doc_wordID_data[pid].keys()
            else:
                wordID_seq = self.doc_wordID_data[pid]
            seq_len = min(len(wordID_seq), self.max_seq_len)
            self.doc_length[pid] = seq_len
            padding_len = self.max_seq_len - seq_len
            x = np.array(wordID_seq, dtype=int) + 1
            if padding_len > 0:
                x = np.concatenate((x, np.zeros(padding_len)))
            self.doc_wordID_data[pid] = x[:self.max_seq_len]
            # labels
            candidate_label, count_score = zip(*(self.candidate_label_data[pid].iteritems()))
            self.candidate_label[pid] = candidate_label
            self.candidate_count_score[pid] = count_score
            true_labels = self.label_data[pid]
            # candidate label for validation
            self.candidate_label_embedding_id[pid] = np.zeros(len(candidate_label))
            self.candidate_label_y[pid] = np.zeros(len(candidate_label))
            for j in range(len(candidate_label)):
                l_ = candidate_label[j]
                self.candidate_label_embedding_id[pid][j] = self.label_dict[l_]
                if l_ in true_labels:
                    self.candidate_label_y[pid][j] = 1

    def get_one_pid_x(self, pid):
        candidate_labels = self.candidate_label[pid]
        num = len(candidate_labels)
        #batch_pid = [pid] * num
        batch_x = [self.doc_wordID_data[pid]] * num
        batch_y = self.candidate_label_y[pid]
        batch_length = [self.doc_length[pid]] * num
        batch_label_embedding_id = self.candidate_label_embedding_id[pid]
        batch_label_prop = [self.label_prop[e] for e in candidate_labels]
        batch_count_score = self.candidate_count_score[pid]
        return pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop, batch_count_score

    def get_pid_x(self, length, start, end):
        batch_pid = []
        batch_x = []
        batch_y = []
        batch_length = []
        batch_label_embedding_id = []
        batch_label_prop = []
        batch_count_score = []
        end2 = min(length, end)
        for i in xrange(start, end2):
            pid = self.pids[i]
            candidate_labels = self.candidate_label[pid]
            num = len(candidate_labels)
            batch_pid.append([pid] * num)
            batch_x.append([self.doc_wordID_data[pid]] * num)
            batch_y.append(self.candidate_label_y[pid])
            batch_length.append([self.doc_length[pid]] * num)
            batch_label_embedding_id.append(self.candidate_label_embedding_id[pid])
            batch_label_prop.append([self.label_prop[e] for e in candidate_labels])
            batch_count_score.append(self.candidate_count_score[pid])
        if end2 < end:
            batch_pid = np.concatenate(batch_pid, axis=0)
            padding_num = end - end2
            batch_x = np.concatenate((np.concatenate(batch_x, axis=0),
                                      np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.concatenate(batch_y, axis=0),
                                      np.zeros(padding_num)), axis=0)
            batch_length = np.concatenate((np.concatenate(batch_length, axis=0),
                                           np.zeros(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.concatenate(batch_label_embedding_id, axis=0),
                                                       np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.concatenate(batch_label_prop, axis=0),
                                               np.zeros(padding_num)), axis=0)
            batch_count_score = np.concatenate((np.concatenate(batch_count_score, axis=0),
                                                np.zeros(padding_num)), axis=0)
        else:
            batch_pid = np.concatenate(batch_pid, axis=0)
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            batch_length = np.concatenate(batch_length, axis=0)
            batch_label_embedding_id = np.concatenate(batch_label_embedding_id, axis=0)
            batch_label_prop = np.concatenate(batch_label_prop, axis=0)
            batch_count_score = np.concatenate(batch_count_score, axis=0)
        return batch_pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop, batch_count_score

    def get_batch(self, length, start, end):
        end2 = min(length, end)
        pid_label_y = self.pid_label_y[start:end2]
        batch_pid = pid_label_y[:, 0]
        batch_label = pid_label_y[:, 1]
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[e] for e in batch_label]
        batch_x = [self.doc_wordID_data[p] for p in batch_pid]
        batch_y = pid_label_y[:, 2]
        batch_label_prop = [self.label_prop[e] for e in batch_label]
        batch_count_score = pid_label_y[:, 3]
        if end2 < end:
            padding_num = end - end2
            batch_length = np.concatenate((np.array(batch_length),
                                           np.ones(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.array(batch_label_embedding_id),
                                                       np.zeros(padding_num)), axis=0)
            batch_x = np.concatenate((np.array(batch_x),
                                      np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.array(batch_y),
                                      np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.array(batch_label_prop),
                                               np.zeros(padding_num)), axis=0)
            batch_count_score = np.concatenate((np.array(batch_count_score),
                                                np.zeros(padding_num)), axis=0)
        return batch_pid, batch_x, batch_y, batch_length, batch_label_embedding_id, batch_label_prop, batch_count_score

    def set_all_batch(self):
        self.pid_label_y = []
        for pid in self.pids:
            candidate_labels = self.candidate_label[pid]
            pids = [pid] * len(candidate_labels)
            y = self.candidate_label_y[pid]
            score = self.candidate_count_score[pid]
            self.pid_label_y.append(np.transpose(np.array([pids, candidate_labels, y, score])))
        self.pid_label_y = np.concatenate(self.pid_label_y, axis=0)

    def get_all_metrics_for_baseline_result(self):
        tar_pid_y = {}
        tar_pid_true_label_prop = {}
        pre_pid_score = {}
        pre_pid_prop = {}
        for pid in self.pids:
            tar_pid_y[pid] = self.candidate_label_y[pid]
            tar_pid_true_label_prop[pid] = [self.label_prop[q] for q in self.label_data[pid]]
            pre_pid_prop[pid] = [self.label_prop[e] for e in self.candidate_label[pid]]
            pre_pid_score[pid] = np.array(self.candidate_count_score[pid])
        results = results_for_score_vector(tar_pid_true_label_prop, tar_pid_y, pre_pid_score, pre_pid_prop)
        print '=========== metrics of candidate baseline result =============='
        print results

# DataLoader for graph
class DataLoader_graph():
    def __init__(self, graph, label_dict, neg_samp=10, g_batch_size=10, g_sample_size=64, g_window_size=3, g_path_size=10):
        self.graph = graph
        self.label_dict = label_dict
        self.neg_sample = neg_samp
        self.g_batch_size = g_batch_size
        self.g_sample_size = g_sample_size
        self.g_window_size = g_window_size
        self.g_path_size = g_path_size
        # self.num_ver = max(self.graph.keys()) + 1
        # print self.num_ver
        self.num_ver = len(self.graph.keys())
        print self.num_ver
        # create index_label
        self.index_label = dict((v, k) for k, v in label_dict.items())
        # reset data
        self.reset_data()

    def gen_graph_context(self):
        gl1, gl2, gy = [], [], []
        end = min(self.num_ver, self.cursor + self.g_batch_size)
        for k in self.ind[self.cursor:end]:
            if len(self.graph[self.index_label[k]]) == 0:
                continue
            path = [k]
            for _ in range(self.g_path_size):
                path.append(self.label_dict[random.choice(self.graph[self.index_label[path[-1]]])])
            for l in range(len(path)):
                for m in range(l - self.g_window_size, l + self.g_window_size):
                    if m < 0 or m >= len(path):
                        continue
                    gl1.append(path[l])
                    gl2.append(path[m])
                    gy.append(1.0)
                    for _ in range(self.neg_sample):
                        gl1.append(path[l])
                        gl2.append(random.randint(0, self.num_ver - 1))
                        gy.append(-1.0)
        self.cursor = end
        if self.cursor == self.num_ver:
            self.reset_data()
        return gl1, gl2, gy

    def reset_data(self):
        self.ind = np.random.permutation(self.num_ver)
        self.cursor = 0


class TrainDataLoader_final():
    def __init__(self, doc_wordID_data, label_data,
                 feature_processor,
                 candidate_label_data,
                 label_dict, label_prop,
                 num_pos=16, num_neg=16,
                 max_seq_len=3000,
                 ):
        self.doc_wordID_data = doc_wordID_data
        self.x_feature_indices = {}
        self.x_feature_values = {}
        self.label_data = label_data
        self.feature_processor = feature_processor
        self.candidate_label_data = candidate_label_data
        self.doc_length = {}
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.label_prop = label_prop
        self.all_labels = label_dict.keys()
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.max_seq_len = max_seq_len
        self.label_pos_pid = {}
        self.label_neg_pid = {}
        self.candidate_label = {}
        self.candidate_count_score = {}
        self.candidate_nlabel_embedding_id = {}
        self.candidate_nlabel_y = {}
        # self.train_pids = []
        self.val_pids = []
        self.label_pid_y = []
        self.initialize_dataloader()
        self.reset_data()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        for pid in self.pids:
            # doc_wordID_data
            temp = sorted(self.doc_wordID_data[pid].items(), key=lambda e: e[1], reverse=True)
            temp2 = sorted(temp[:self.max_seq_len], key=lambda e: e[0], reverse=False)
            feature_id, feature_v = zip(*temp2)
            seq_len = min(len(feature_id), self.max_seq_len)
            feature_indices = np.array(list(feature_id) + (self.max_seq_len-seq_len)*[0])
            feature_indices[:seq_len] = feature_indices[:seq_len] + 1
            self.x_feature_indices[pid] = feature_indices
            self.x_feature_values[pid] = np.array(list(feature_v) + (self.max_seq_len-seq_len)*[0])
            self.doc_length[pid] = seq_len
            # labels
            candidate_label, count_score = zip(*(self.candidate_label_data[pid].iteritems()))
            self.candidate_label[pid] = candidate_label
            self.candidate_count_score[pid] = count_score
            pos_labels = self.label_data[pid]
            neg_labels = list(set(candidate_label) - set(pos_labels))
            for label in pos_labels:
                try:
                    self.label_pos_pid[label].append(pid)
                    self.feature_processor.label_pool_feature[label] = np.union1d(self.feature_processor.label_pool_feature, feature_id)
                except KeyError:
                    self.label_pos_pid[label] = [pid]
                    self.feature_processor.label_pool_feature[label] = feature_id
            for label in neg_labels:
                try:
                    self.label_neg_pid[label].append(pid)
                except KeyError:
                    self.label_neg_pid[label] = [pid]
            # candidate label for validation
            self.candidate_nlabel_embedding_id[pid] = np.zeros(len(candidate_label))
            self.candidate_nlabel_y[pid] = np.zeros(len(candidate_label))
            for j in range(len(candidate_label)):
                l_ = candidate_label[j]
                self.candidate_nlabel_embedding_id[pid][j] = self.label_dict[l_]
                if l_ in pos_labels:
                    self.candidate_nlabel_y[pid][j] = 1

    def set_val_batch(self):
        self.val_pid_label_y = []
        for pid in self.val_pids:
            candidate_labels = self.candidate_label[pid]
            pids = [pid] * len(candidate_labels)
            y = self.candidate_nlabel_y[pid]
            score = self.candidate_count_score[pid]
            self.val_pid_label_y.append(np.transpose(np.array([pids, candidate_labels, y, score])))
        self.val_pid_label_y = np.concatenate(self.val_pid_label_y, axis=0)

    def get_val_batch(self, length, start, end):
        end2 = min(length, end)
        pid_label_y = self.val_pid_label_y[start:end2]
        batch_pid = pid_label_y[:, 0]
        batch_label = pid_label_y[:, 1]
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[e] for e in batch_label]
        #batch_x = [self.doc_wordID_data[p] for p in batch_pid]
        batch_x_feature_id = [self.x_feature_indices[p] for p in batch_pid]
        batch_x_feature_v = [self.x_feature_values[p] for p in batch_pid]
        batch_y = pid_label_y[:, 2]
        batch_label_prop = [self.label_prop[e] for e in batch_label]
        batch_count_score = pid_label_y[:, 3]
        lbl_active_fea_id = [self.feature_processor.label_active_feature_ids[lbl_idx] for lbl_idx in
                             batch_label]
        if end2 < end:
            padding_num = end - end2
            batch_length = np.concatenate((np.array(batch_length),
                                           np.ones(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.array(batch_label_embedding_id),
                                                       np.zeros(padding_num)), axis=0)
            # batch_x = np.concatenate((np.array(batch_x),
            #                           np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_x_feature_id = np.concatenate((np.array(batch_x_feature_id),
                                                 np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_x_feature_v = np.concatenate((np.array(batch_x_feature_v),
                                                np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.array(batch_y),
                                      np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.array(batch_label_prop),
                                               np.zeros(padding_num)), axis=0)
            batch_count_score = np.concatenate((np.array(batch_count_score),
                                                np.zeros(padding_num)), axis=0)
            lbl_active_fea_id = np.concatenate((np.array(lbl_active_fea_id),
                                                np.zeros((padding_num, self.feature_processor.active_feature_num))), axis=0)
        return lbl_active_fea_id, batch_pid, batch_x_feature_id, batch_x_feature_v, batch_y, batch_length, batch_label_embedding_id, batch_label_prop, batch_count_score

    def next_batch(self, label):
        pid_label_y = self.label_pid_y[label]
        batch_pid = pid_label_y[:, 0]
        batch_y = pid_label_y[:, 1]
        #batch_label = [label] * len(batch_pid)
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[label]] * len(batch_pid)
        batch_x_feature_id = [self.x_feature_indices[p] for p in batch_pid]
        batch_x_feature_v = [self.x_feature_values[p] for p in batch_pid]
        batch_label_prop = [self.label_prop[label]] * len(batch_pid)
        return batch_x_feature_id, batch_x_feature_v, batch_y, batch_length, batch_label_embedding_id, batch_label_prop

    def get_fixed_length_pos_samples(self, items, num):
        length = len(items)
        if length < num:
            sample = items
            pad_sample = np.random.choice(items, num-length)
            sample = sample + pad_sample.tolist()
        elif length > num:
            sample = random.sample(items, num)
        else:
            sample = items
        return sample

    def get_fixed_length_neg_samples(self, label, num):
        try:
            items = self.label_neg_pid[label]
            length = len(items)
            if length < num:
                sample = items
                pad_sample = np.random.choice(items, num-length)
                sample = sample + pad_sample.tolist()
            elif length > num:
                sample = random.sample(items, num)
            else:
                sample = items
        except KeyError:
            neg_set = list(set(self.pids) - set(self.label_pos_pid[label]))
            sample = np.random.choice(neg_set, num).tolist()
        return sample

    def reset_data(self):
        self.label_pid_y = {}
        for label in self.all_labels:
            pos_pid = self.get_fixed_length_pos_samples(self.label_pos_pid[label], self.num_pos)
            neg_pid = self.get_fixed_length_neg_samples(label, self.num_neg)
            pids = pos_pid + neg_pid
            y = [1] * self.num_pos + [0] * self.num_neg
            self.label_pid_y[label] = np.transpose(np.array([pids, y]))
        _, self.val_pids = train_test_split(self.pids, test_size=0.1)
        self.set_val_batch()

class TestDataLoader_final():
    def __init__(self, doc_wordID_data, label_data,
                 feature_processor,
                 candidate_label_data,
                 label_dict, label_prop,
                 max_seq_len=3000,
                 ):
        self.doc_wordID_data = doc_wordID_data
        self.x_feature_indices = {}
        self.x_feature_values = {}
        self.feature_processor = feature_processor
        self.label_data = label_data
        self.candidate_label_data = candidate_label_data
        self.doc_length = {}
        self.pids = self.label_data.keys()
        self.label_dict = label_dict
        self.label_prop = label_prop
        self.all_labels = label_dict.keys()
        self.max_seq_len = max_seq_len
        self.candidate_label = {}
        self.candidate_count_score = {}
        self.candidate_nlabel_embedding_id = {}
        self.candidate_nlabel_y = {}
        self.pid_label_y = []
        self.initialize_dataloader()
        self.reset_data()

    def initialize_dataloader(self):
        print 'num of doc:             ' + str(len(self.doc_wordID_data))
        print 'num of y:               ' + str(len(self.label_data))
        print 'num of candidate_label: ' + str(len(self.candidate_label_data))
        # doc_token_data consists of wordIDs in vocab.
        for pid in self.pids:
            # doc_wordID_data
            temp = sorted(self.doc_wordID_data[pid].items(), key=lambda e: e[1], reverse=True)
            temp2 = sorted(temp[:self.max_seq_len], key=lambda e: e[0], reverse=False)
            feature_id, feature_v = zip(*temp2)
            seq_len = min(len(feature_id), self.max_seq_len)
            feature_indices = np.array(list(feature_id) + (self.max_seq_len-seq_len)*[0])
            feature_indices[:seq_len] = feature_indices[:seq_len] + 1
            self.x_feature_indices[pid] = feature_indices
            self.x_feature_values[pid] = np.array(list(feature_v) + (self.max_seq_len-seq_len)*[0])
            self.doc_length[pid] = seq_len
            # labels
            candidate_label, count_score = zip(*(self.candidate_label_data[pid].iteritems()))
            self.candidate_label[pid] = candidate_label
            self.candidate_count_score[pid] = count_score
            pos_labels = self.label_data[pid]
            # candidate label for validation
            self.candidate_nlabel_embedding_id[pid] = np.zeros(len(candidate_label))
            self.candidate_nlabel_y[pid] = np.zeros(len(candidate_label))
            for j in range(len(candidate_label)):
                l_ = candidate_label[j]
                self.candidate_nlabel_embedding_id[pid][j] = self.label_dict[l_]
                if l_ in pos_labels:
                    self.candidate_nlabel_y[pid][j] = 1

    def get_batch(self, length, start, end):
        end2 = min(length, end)
        pid_label_y = self.pid_label_y[start:end2]
        batch_pid = pid_label_y[:, 0]
        batch_label = pid_label_y[:, 1]
        batch_length = [self.doc_length[p] for p in batch_pid]
        batch_label_embedding_id = [self.label_dict[e] for e in batch_label]
        batch_x_feature_id = [self.x_feature_indices[p] for p in batch_pid]
        batch_x_feature_v = [self.x_feature_values[p] for p in batch_pid]
        batch_y = pid_label_y[:, 2]
        batch_label_prop = [self.label_prop[e] for e in batch_label]
        batch_count_score = pid_label_y[:, 3]
        lbl_active_fea_id = [self.feature_processor.label_active_feature_ids[lbl_idx] for lbl_idx in
                             batch_label]
        if end2 < end:
            padding_num = end - end2
            batch_length = np.concatenate((np.array(batch_length),
                                           np.ones(padding_num)), axis=0)
            batch_label_embedding_id = np.concatenate((np.array(batch_label_embedding_id),
                                                       np.zeros(padding_num)), axis=0)
            # batch_x = np.concatenate((np.array(batch_x),
            #                           np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_x_feature_id = np.concatenate((np.array(batch_x_feature_id),
                                                 np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_x_feature_v = np.concatenate((np.array(batch_x_feature_v),
                                                np.zeros((padding_num, self.max_seq_len))), axis=0)
            batch_y = np.concatenate((np.array(batch_y),
                                      np.zeros(padding_num)), axis=0)
            batch_label_prop = np.concatenate((np.array(batch_label_prop),
                                               np.zeros(padding_num)), axis=0)
            batch_count_score = np.concatenate((np.array(batch_count_score),
                                                np.zeros(padding_num)), axis=0)
            lbl_active_fea_id = np.concatenate((np.array(lbl_active_fea_id),
                                                np.zeros((padding_num, self.feature_processor.active_feature_num))),
                                               axis=0)
        return lbl_active_fea_id, batch_pid, batch_x_feature_id, batch_x_feature_v, \
               batch_y, batch_length, batch_label_embedding_id, \
               batch_label_prop, batch_count_score

    def reset_data(self):
        self.pid_label_y = []
        for pid in self.pids:
            candidate_labels = self.candidate_label[pid]
            pids = [pid] * len(candidate_labels)
            y = self.candidate_nlabel_y[pid]
            score = self.candidate_count_score[pid]
            self.pid_label_y.append(np.transpose(np.array([pids, candidate_labels, y, score])))
        self.pid_label_y = np.concatenate(self.pid_label_y, axis=0)


class FeatureProcessor():
    def __init__(self, feature_num, active_feature_num=100):
        self.feature_num = feature_num
        self.active_feature_num = active_feature_num
        self.label_pool_feature = {}
        self.label_active_feature_grads = {}
        self.label_active_feature_ids = {}

    def set_active_feature_grads(self, label, word_grads):
        word_grads = np.absolute(word_grads)
        active_grads = word_grads.copy()
        sort_idx = np.argsort(word_grads)[:(self.feature_num-self.active_feature_num)]
        active_grads[sort_idx] = 0
        try:
            self.label_active_feature_grads[label] += active_grads
        except KeyError:
            self.label_active_feature_grads[label] = active_grads

    def set_active_feature_id(self):
        for label, grads in self.label_active_feature_grads.items():
            #print 'max: '
            #print np.max(grads)
            #print 'min: '
            #print np.min(grads)
            idx = np.argsort(-grads)[:self.active_feature_num]
            non_zero_idx = np.nonzero(grads)
            active_feature_id = np.intersect1d(np.intersect1d(idx, non_zero_idx), self.label_pool_feature[label]) + 1
            #print 'number of active features'
            if len(active_feature_id) < self.active_feature_num:
                print len(active_feature_id)
                padding_num = self.active_feature_num - len(active_feature_id)
                active_feature_id = np.concatenate((active_feature_id, np.zeros(padding_num)))
            self.label_active_feature_ids[label] = active_feature_id


