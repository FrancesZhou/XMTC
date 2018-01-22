'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import math
from sklearn.model_selection import train_test_split
import re
import copy
from .preprocessing import generate_embedding_from_vocabID, generate_label_vector_of_fixed_length, get_wordID_from_vocab

class DataLoader2():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data, topk,
                 label_dict,
                 #batch_size,
                 max_seq_len=3000,
                 if_use_all_true_label=0):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.given_candidate_label_data = candidate_label_data
        self.candidate_label_data = {}
        self.candidate_label_embedding_id = {}
        self.candidate_label_prop = {}
        self.doc_length = {}
        self.topk_candidate = topk
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
                    num_candidate = max(self.topk_candidate, num_true)
                    self.candidate_label_data[pid] = np.zeros(num_candidate)
                    self.candidate_label_embedding_id[pid] = np.zeros(num_candidate)
                    self.candidate_label_prop[pid] = np.zeros([num_candidate, 2])
                    self.candidate_label_data[pid][:num_true] = true_labels
                    self.candidate_label_embedding_id[pid][:num_true] = [self.label_dict[e] for e in true_labels]
                    self.candidate_label_prop[pid][:num_true] = np.tile([0, 1], [num_true, 1])
                    if num_true < num_candidate:
                        neg_labels = [e for e in self.given_candidate_label_data[pid] if e not in true_labels]
                        for j in range(num_true, self.topk_candidate):
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
                    self.candidate_label_data[pid] = self.given_candidate_label_data[pid][:self.topk_candidate]
                    self.candidate_label_embedding_id[pid] = np.zeros(self.topk_candidate)
                    self.candidate_label_prop[pid] = np.zeros([self.topk_candidate, 2])
                    for j in range(self.topk_candidate):
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
        # batch_pid = np.zeros([pid_num, self.topk_candidate])
        # batch_length = np.zeros([pid_num, self.topk_candidate])
        # batch_x = np.zeros([pid_num, self.topk_candidate, self.max_seq_len])
        # batch_label = np.zeros([pid_num, self.topk_candidate])
        # batch_label_embedding_id = np.zeros([pid_num, self.topk_candidate])
        # batch_y = np.zeros([pid_num, self.topk_candidate, 2])
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
            # batch_pid[i] = [pid]*self.topk_candidate
            # batch_length[i] = [self.doc_length[pid]]*self.topk_candidate
            # batch_x[i] = np.tile(self.doc_wordID_data[pid], [self.topk_candidate, 1])
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
class DataLoader5():
    def __init__(self, doc_wordID_data, label_data,
                 all_labels,
                 batch_size,
                 given_seq_len=False, max_seq_len=5000):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.pids = self.label_data.keys()
        self.all_labels = all_labels
        self.batch_size = batch_size
        self.given_seq_len = given_seq_len
        self.max_seq_len = max_seq_len
        self.doc_length = {}
        self.label_dict = {}
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        # create label_dict
        self.label_dict = dict(zip(self.all_labels, range(len(self.all_labels))))
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
        self.pids = self.label_data.keys()
        # assign max_seq_len if not given_seq_len
        print 'after removing zero-length data'
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        if not self.given_seq_len:
            self.max_seq_len = min(max(all_length), self.max_seq_len)
        print 'max sequence length: ' + str(self.max_seq_len)

    def get_pid_x(self, i, j):
        batch_pid = []
        batch_x = []
        batch_y = []
        end = min(j, len(self.pids))
        for pid in self.pids[i:end]:
            batch_pid.append(pid)
            #_, seq_emb = generate_embedding_from_vocabID(self.doc_wordID_data[pid], self.max_seq_len, self.word_embeddings)
            #seq_len = min(self.doc_length[pid], self.max_seq_len)
            padding_len = self.max_seq_len - min(self.doc_length[pid], self.max_seq_len)
            x = np.array(self.doc_wordID_data[pid], dtype=int) + 1
            x = x.tolist()
            if padding_len:
                x = x + [0] * padding_len
            x = x[:self.max_seq_len]
            #
            y = np.zeros(len(self.all_labels))
            for l in self.label_data[pid]:
                y[self.label_dict[l]] = 1
            batch_x.append(x)
            batch_y.append(y)
        if end < j:
            batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
            batch_y = np.concatenate((batch_y, np.zeros((j-end, len(self.all_labels)))), axis=0)
        # while end < j:
        #     batch_x.append(np.zeros((self.max_seq_len, self.word_embeddings.shape[-1])))
        #     batch_y.append(np.zeros((len(self.all_labels))))
        #     #print len(batch_x)
        #     end += 1
        return batch_pid, batch_x, batch_y
