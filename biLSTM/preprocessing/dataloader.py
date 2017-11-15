'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
from .preprocessing import generate_embedding_from_vocabID, generate_label_vector, gen_word_emb_from_str

class DataLoader():
    def __init__(self, data, labels, batch_size, max_seq_len, num_labels, word_embeddings):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.word_embeddings = word_embeddings
        self.num_labels = num_labels
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
        self.batch_start = range(self.num_batch) * self.batch_size
        self.batch_end = range(1, self.num_batch) * self.batch_size

    def next_batch(self):
        data_batch = self.data[self.batch_start[self.pointer]:self.batch_end[self.pointer]]
        labels_batch = self.labels[self.batch_start[self.pointer]:self.batch_end[self.pointer]]
        #
        batch_x = []
        batch_y = []
        batch_l = []
        for s in range(len(data_batch)):
            try:
                seq_len, emb = generate_embedding_from_vocabID(data_batch[s], self.max_seq_len, self.word_embeddings)
            except Exception as e:
                print s
                raise e
            l_v = generate_label_vector(labels_batch[s], self.num_labels)
            batch_x.append(emb)
            batch_y.append(l_v)
            batch_l.append(seq_len)

        self.pointer = (self.pointer+1) % self.num_batch
        return batch_x, batch_y, batch_l

    def reset_pointer(self):
        self.pointer = 0