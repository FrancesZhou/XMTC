'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import re
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
    def __init__(self, doc_data, label_data, all_labels, label_embeddings, batch_size, vocab, word_embeddings, pos_neg_ratio, max_seq_len=None):
        self.doc_data = doc_data
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
        print 'num of doc: ' + str(len(self.doc_data))
        print 'num of y: ' + str(len(self.label_data))
        # define number of positive and negative samples in a batch
        self.num_pos = self.batch_size / (self.pos_neg_ratio + 1)
        self.num_neg = self.batch_size - self.num_pos
        # doc_token_data consists of wordIDs in vocab.
        self.doc_token_data = {}
        self.doc_length = {}
        all_length = []
        for pid, seq in self.doc_data.items():
            token_indices = get_wordID_from_vocab(seq, self.vocab)
            self.doc_token_data[pid] = token_indices
            all_length.append(len(token_indices))
            self.doc_length[pid] = len(token_indices)
        # assign max_seq_len if None
        if self.max_seq_len is None:
            self.max_seq_len = max(all_length)
        self.reset_data()

    def generate_pos_sample(self):
        pid = np.random.choice(self.pids_copy)
        label = np.random.choice(self.label_data_copy[pid])
        # follow-up processing
        self.label_data_copy[pid].remove(label)
        if ~self.label_data_copy[pid]:
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
            _, embeddings = generate_embedding_from_vocabID(self.doc_token_data[pid], self.max_seq_len, self.word_embeddings)
            batch_x.append(embeddings)
            batch_y.append([0, 1])
            batch_length.append(self.doc_length[pid])
            batch_label_embedding.append(self.word_embeddings[label])
            if len(self.pids_copy) == 0:
                self.end_of_data = True
                break
        # negative
        for i in range(self.num_neg):
            pid, label = self.generate_neg_sample()
            batch_pid.append(pid)
            batch_label.append(label)
            _, embeddings = generate_embedding_from_vocabID(self.doc_token_data[pid], self.max_seq_len, self.word_embeddings)
            batch_x.append(embeddings)
            batch_y.append([1, 0])
            batch_length.append(self.doc_length[pid])
            batch_label_embedding.append(self.word_embeddings[label])
        return batch_x, batch_y, batch_length, batch_label_embedding

    def reset_data(self):
        self.label_data_copy = dict(self.label_data)
        self.pids_copy = list(self.pids)
        self.end_of_data = False
