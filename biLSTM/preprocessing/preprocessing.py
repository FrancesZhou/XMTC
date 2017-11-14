'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import re
import numpy as np
from ..utils.io_utils import dump_json, load_json, dump_pickle, load_pickle

def not_empty(s):
    #return s.strip()
    return s and s.strip()

def construct_corpus_from_file(corpus_path):
    corpus = []
    try:
        fp = open(corpus_path, 'r')
        for line in open(corpus_path):
            line = fp.readline()
            text = line.split('->', 1)[1]
            all_tokens = re.split('([A-Za-z]+)|([0-9]+)', text)
            #print all_tokens
            all_tokens = filter(not_empty, all_tokens)
            all_tokens = [e.strip() for e in all_tokens]
            corpus.append(all_tokens)

        fp.close()
    except Exception as e:
        raise e
    return corpus

def construct_train_test_corpus(train_path, test_path, output):
    train_corpus = construct_corpus_from_file(train_path)
    test_corpus = construct_corpus_from_file(test_path)
    dump_pickle(train_corpus, os.path.join(output, 'train.corpus'))
    dump_pickle(test_corpus, os.path.join(output, 'test.corpus'))
    return train_corpus, test_corpus

def generate_labels_from_file(file_name, output):
    labels = []
    try:
        fp = open(file_name, 'r')
        header = fp.readline()
        while True:
            line = fp.readline()
            if not line:
                break
            labels_str = line.split(' ', 1)[0]
            labels_str = labels_str.split(',')
            labels_doc = [int(label) for label in labels_str]
            labels.append(labels_doc)
    except Exception as e:
        raise e
    dump_pickle(labels, output)
    return labels

def generate_label_pair_from_file(file_name, output):
    labels = load_pickle(file_name)
    label_pairs = []
    for labels_doc in labels:
        if len(labels_doc) == 1:
            continue
        labels_doc = sorted(labels_doc)
        label_pair_start = labels_doc[0]
        for label in labels_doc[1:]:
            label_pairs.append([label_pair_start, label])
    # delete duplica
    label_pairs = np.array(label_pairs, dtype=np.int64)
    label_pairs = np.unique(label_pairs, axis=0)
    dump_pickle(label_pairs, output)
    return label_pairs

def get_max_seq_len(data):
    num = len(data)
    all_seq_len = np.zeros(num)
    for i in range(num):
        all_seq_len[i] = len(data[i])
    max_seq_len = max(all_seq_len)
    return max_seq_len

def batch_data(data, labels, max_seq_len, num_label, vocab, word_embeddings, batch_size=32):
    num = len(data)
    #max_seq_len = get_max_seq_len(data)
    x = []
    y = []
    length = []
    i = 0
    while i < num:
        if i % 10000 == 0:
            print i
        batch_x = []
        batch_y = []
        batch_l = []
        # --- x
        for s in range(i, max(i+batch_size, num)):
            seq_len, emb = generate_embedding(data[s], max_seq_len, vocab, word_embeddings)
            l_v = generate_label_vector(labels[s], num_label)
            batch_x.append(emb)
            batch_y.append(l_v)
            batch_l.append(seq_len)
        x.append(batch_x)
        y.append(batch_y)
        length.append(batch_l)
        i = i + batch_size
    return x, y, length

def generate_embedding(sequence, max_seq_len, vocab, word_embeddings):
    embeddings = []
    seq_len = min(len(sequence), max_seq_len)
    for i in range(seq_len):
        index = vocab.index(sequence[i])
        emb_str = word_embeddings[index]
        embeddings.append(gen_word_emb_from_str(emb_str))
    zero_len = max_seq_len - seq_len
    embeddings = np.array(embeddings)
    if zero_len>0:
        zero_emb = np.zeros([zero_len, embeddings.shape[-1]])
        embeddings = np.concatenate((embeddings, zero_emb), axis=0)
    return seq_len, embeddings

def gen_word_emb_from_str(str):
    _, v_str = str.split(' ', 1)
    v = [float(e) for e in v_str.split()]
    return v

def generate_label_vector(labels, num_label):
    return np.sum(np.eye(num_label)[labels], axis=0)


