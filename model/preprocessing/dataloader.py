'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import re
import copy
from .preprocessing import generate_embedding_from_vocabID, generate_label_vector_of_fixed_length, get_wordID_from_vocab

class DataLoader():
    def __init__(self, data, labels, batch_size, max_seq_len, num_labels, num_all_labels, word_embeddings):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.word_embeddings = word_embeddings
        self.num_labels = num_labels
        self.num_all_labels = num_all_labels
        self.create_batches()
        self.reset_pointer()

    def create_batches(self):
        print 'num of data: ' + str(len(self.data))
        print 'num of y: ' + str(len(self.labels))
        self.num_batch = int(len(self.data)/self.batch_size)
        if len(self.data) == len(self.labels):
            self.num_of_used_data = self.num_batch * self.batch_size
        else:
            print 'error: num of data is not equal to num of y.'
        self.batch_start = np.arange(self.num_batch) * self.batch_size
        self.batch_end = np.arange(1, self.num_batch+1) * self.batch_size

    def next_batch(self):
        data_batch = self.data[self.batch_start[self.pointer]:self.batch_end[self.pointer]]
        labels_batch = self.labels[self.batch_start[self.pointer]:self.batch_end[self.pointer]]
        #
	    #print len(data_batch)
	    #print len(labels_batch)
        batch_x = []
        batch_y = []
        batch_l = []
        batch_i = []
        for s in range(len(data_batch)):
            try:
                seq_len, emb = generate_embedding_from_vocabID(data_batch[s], self.max_seq_len, self.word_embeddings)
            except Exception as e:
                print s
                raise e
            l_indices, l_values = generate_label_vector_of_fixed_length(labels_batch[s], self.num_labels, self.num_all_labels)
            batch_x.append(emb)
            batch_y.append(l_values)
            batch_l.append(seq_len)
            batch_i.append(l_indices)

        self.pointer = (self.pointer+1) % self.num_batch
        return batch_x, batch_y, batch_l, batch_i

    def reset_pointer(self):
        self.pointer = 0


class DataLoader2():
    def __init__(self, doc_wordID_data, label_data, all_labels, label_embeddings, batch_size, vocab, word_embeddings, pos_neg_ratio, max_seq_len=None):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.pids = self.label_data.keys()
        self.all_labels = all_labels
        self.label_embeddings = label_embeddings
        self.batch_size = batch_size
        self.vocab = vocab
        self.word_embeddings = word_embeddings
        self.pos_neg_ratio = pos_neg_ratio
        self.max_seq_len = max_seq_len
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        # define number of positive and negative samples in a batch
        self.num_pos = self.batch_size / (self.pos_neg_ratio + 1)
        self.num_neg = self.batch_size - self.num_pos
        # doc_token_data consists of wordIDs in vocab.
        self.doc_length = {}
        all_length = []
        count = 0
        for pid, seq in self.doc_wordID_data.items():
            count += 1
            if count % 50 == 0:
                print count
            all_length.append(len(seq))
            self.doc_length[pid] = len(seq)
        # assign max_seq_len if None
        if self.max_seq_len is None:
            self.max_seq_len = max(all_length)
        self.reset_data()

    def generate_pos_sample(self):
        pid = np.random.choice(self.pids_copy)
        label = np.random.choice(self.label_data_copy[pid])
        # follow-up processing
        self.label_data_copy[pid].remove(label)
        if not self.label_data_copy[pid]:
            self.pids_copy.remove(pid)
            del self.label_data_copy[pid]
        return pid, label

    def generate_neg_sample(self):
        pid = np.random.choice(self.pids)
        label = np.random.choice(list(set(self.all_labels) - set(self.label_data[pid])))
        return pid, label

    def next_batch(self):
        batch_pid = []
        batch_label = []
        batch_x = []
        batch_y = []
        batch_length = []
        batch_label_embedding = []
        # positive
        for i in range(self.num_pos):
            pid, label = self.generate_pos_sample()
            batch_pid.append(pid)
            batch_label.append(label)
            _, embeddings = generate_embedding_from_vocabID(self.doc_wordID_data[pid], self.max_seq_len, self.word_embeddings)
            batch_x.append(embeddings)
            batch_y.append([0, 1])
            batch_length.append(self.doc_length[pid])
            batch_label_embedding.append(self.label_embeddings[label])
            if not self.pids_copy:
                self.end_of_data = True
                break
        # negative
        for i in range(self.num_neg):
            pid, label = self.generate_neg_sample()
            batch_pid.append(pid)
            batch_label.append(label)
            _, embeddings = generate_embedding_from_vocabID(self.doc_wordID_data[pid], self.max_seq_len, self.word_embeddings)
            batch_x.append(embeddings)
            batch_y.append([1, 0])
            batch_length.append(self.doc_length[pid])
            batch_label_embedding.append(self.label_embeddings[label])
        return batch_pid, batch_label, batch_x, batch_y, batch_length, batch_label_embedding

    def reset_data(self):
        # self.label_data_copy = dict(self.label_data)
        # self.pids_copy = list(self.pids)
        self.label_data_copy = copy.deepcopy(self.label_data)
        self.pids_copy = copy.deepcopy(self.pids)
        self.end_of_data = False

# DataLoader3 is for testing all texts for all labels
# produce each text and all labels per next_text()
class DataLoader3():
    def __init__(self, doc_wordID_data, label_data, all_labels, label_embeddings, batch_size, vocab, word_embeddings, max_seq_len=None):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.pids = self.label_data.keys()
        self.all_labels = all_labels
        self.label_embeddings = label_embeddings
        self.batch_size = batch_size
        self.vocab = vocab
        self.word_embeddings = word_embeddings
        self.max_seq_len = max_seq_len
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        self.num_batch_for_each_text = len(self.all_labels)/self.batch_size + 1
        self.samples_of_last_batch_for_each_text = len(self.all_labels) % self.batch_size
        # set all_label_embeddings for all_labels
        self.all_label_embedding = []
        for label in self.all_labels:
            self.all_label_embedding.append(self.label_embeddings[label])
        # set doc_length according to doc_wordID_data
        # doc_wordID_data consists of wordIDs in vocab.
        self.doc_length = {}
        count = 0
        for pid, seq in self.doc_wordID_data.items():
            count += 1
            if count % 50 == 0:
                print count
            self.doc_length[pid] = len(seq)
        # assign max_seq_len if None
        if self.max_seq_len is None:
            self.max_seq_len = max(self.doc_length.values())
        self.reset_data()

    def next_text(self):
        all_label_embedding = np.array(self.all_label_embedding)
        try:
            pid = self.pids_copy.pop()
            _, x = generate_embedding_from_vocabID(self.doc_wordID_data[pid], self.max_seq_len, self.word_embeddings)
            x = np.tile(x, (self.batch_size, 1, 1))
            #print x.shape
            seq_len = np.repeat(self.doc_length[pid], self.batch_size)
            #print seq_len.shape
            all_y = []
            for label in self.all_labels:
                if label in self.label_data[pid]:
                    all_y.append(1)
                else:
                    all_y.append(0)
            if self.samples_of_last_batch_for_each_text:
                all_y = np.concatenate((all_y, np.zeros(self.batch_size-self.samples_of_last_batch_for_each_text)))
                all_label_embedding = np.concatenate((all_label_embedding,
                                                      np.zeros((self.batch_size-self.samples_of_last_batch_for_each_text, all_label_embedding.shape[-1]))), axis=0)
        except IndexError:
            self.end_of_data = True
            return 0, 0, 0, 0, 0, 0
        all_y = np.expand_dims(np.array(all_y), axis=-1)
        all_y = np.concatenate((all_y, 1-all_y), axis=-1)
        all_y = np.stack(np.split(np.array(all_y), self.num_batch_for_each_text, axis=0))
        all_label_embedding = np.stack(np.split(np.array(all_label_embedding), self.num_batch_for_each_text, axis=0))
        return pid, x, seq_len, self.all_labels, all_y, all_label_embedding

    def reset_data(self):
        #self.label_data_copy = copy.deepcopy(self.label_data)
        self.pids_copy = copy.deepcopy(self.pids)
        self.end_of_data = False

# DataLoader4 is for loading candidate label subset from SLEEC
class DataLoader4():
    def __init__(self, doc_wordID_data, label_data,
                 candidate_label_data,
                 all_labels, label_embeddings,
                 batch_size,
                 vocab, word_embeddings,
                 given_seq_len=False, max_seq_len=5000,
                 if_use_all_true_label=0):
        self.doc_wordID_data = doc_wordID_data
        self.label_data = label_data
        self.pids = self.label_data.keys()
        self.all_labels = all_labels
        self.candidate_label_data = candidate_label_data
        self.label_embeddings = label_embeddings
        self.batch_size = batch_size
        self.vocab = vocab
        self.word_embeddings = word_embeddings
        self.given_seq_len = given_seq_len
        self.max_seq_len = max_seq_len
        self.if_use_all_true_label = if_use_all_true_label
        self.initialize_dataloader()

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        # doc_token_data consists of wordIDs in vocab.
        self.doc_length = {}
        all_length = []
        for pid, seq in self.doc_wordID_data.items():
            all_length.append(len(seq))
            self.doc_length[pid] = len(seq)
        # assign max_seq_len if not given_seq_len
        if not self.given_seq_len:
            self.max_seq_len = min(max(all_length), self.max_seq_len)
        # if_use_all_true_label
        if self.if_use_all_true_label:
            for pid, label in self.label_data.items():
                candidate_label = self.candidate_label_data[pid]
                #candidate_label = filter(lambda x: x != 0, candidate_label)
                #candidate_label = np.asarray(candidate_label) - 1
                candidate_label = list(set(candidate_label) & set(self.all_labels))
                self.candidate_label_data[pid] = np.unique(np.concatenate((candidate_label, label))).tolist()
        else:
            for pid, candidate_label in self.candidate_label_data.items():
                #candidate_label = filter(lambda x: x != 0, self.candidate_label_data[pid])
                #candidate_label = np.asarray(candidate_label) - 1
                candidate_label = list(set(candidate_label) & set(self.all_labels))
                self.candidate_label_data[pid] = candidate_label
        self.reset_data()

    def get_pid_x(self, i, j):
        batch_pid = []
        batch_x = []
        batch_length = []
        end = min(j, len(self.pids))
        for pid in self.pids[i:end]:
            batch_pid.append(pid)
            seq_len, seq_emb = generate_embedding_from_vocabID(self.doc_wordID_data[pid], self.max_seq_len, self.word_embeddings)
            batch_length.append(seq_len)
            batch_x.append(seq_emb)
        while end < j:
            batch_length.append([int(0)])
            batch_x.append(np.zeros((self.max_seq_len, self.word_embeddings.shape[-1])))
            end += 1
        return batch_pid, batch_x, batch_length

    def generate_sample(self):
        pid = np.random.choice(self.pids_copy)
        label = np.random.choice(self.candidate_label_data_copy[pid])
        # follow-up processing
        self.candidate_label_data_copy[pid].remove(label)
        if not self.candidate_label_data_copy[pid]:
            self.pids_copy.remove(pid)
            del self.candidate_label_data_copy[pid]
        return pid, label

    def next_batch(self):
        batch_pid = []
        batch_label = []
        batch_x = []
        batch_y = []
        batch_length = []
        batch_label_embedding = []
        i = 0
        while i < self.batch_size:
            pid, label = self.generate_sample()
            seq_len, embeddings = generate_embedding_from_vocabID(self.doc_wordID_data[pid], self.max_seq_len, self.word_embeddings)
            if seq_len == 0:
                if not self.pids_copy:
                    self.end_of_data = True
                    break
                else:
                    continue
            batch_pid.append(pid)
            batch_label.append(label)
            batch_x.append(embeddings)
            if label in self.label_data[pid]:
                batch_y.append([0, 1])
            else:
                batch_y.append([1, 0])
            #batch_length.append(self.doc_length[pid])
            batch_length.append(seq_len)
            batch_label_embedding.append(self.label_embeddings[label])
            i = i + 1
            if not self.pids_copy:
                self.end_of_data = True
                break
        return batch_pid, batch_label, batch_x, batch_y, batch_length, batch_label_embedding

    def reset_data(self):
        #self.label_data_copy = copy.deepcopy(self.label_data)
        self.candidate_label_data_copy = copy.deepcopy(self.candidate_label_data)
        self.pids_copy = copy.deepcopy(self.pids)
        self.end_of_data = False
