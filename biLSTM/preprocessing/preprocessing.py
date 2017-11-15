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
            all_tokens = re.split('([A-Za-z]+)|([0-9]+)|(\W)', text)
            #print all_tokens
            all_tokens = filter(not_empty, all_tokens)
            all_tokens = [e.strip() for e in all_tokens]
            corpus.append(all_tokens)

        fp.close()
    except Exception as e:
        raise e
    return corpus

def construct_corpus_from_file_vocab(corpus_path, vocab):
    corpus = []
    line_index = 0
    error_index = []
    try:
        fp = open(corpus_path, 'r')
        for line in open(corpus_path):
            line = fp.readline()
            line_index += 1
            text = line.split('->', 1)[1]
            all_tokens = re.split('([A-Za-z]+)|([0-9]+)|(\W)', text)
            #print all_tokens
            all_tokens = filter(not_empty, all_tokens)
            all_tokens = [e.strip() for e in all_tokens]
            # check if tokens are in the vocab
            token_indices = []
            for t in all_tokens:
                try:
                    ind = vocab.index(t)
                    token_indices.append(ind)
                except:
                    continue
            if len(token_indices):
                corpus.append(token_indices)
            else:
		print line_index
                error_index.append(line_index)

        fp.close()
    except Exception as e:
        raise e
    print len(corpus)
    print len(error_index)
    return corpus, error_index

def construct_train_test_corpus(vocab, train_path, test_path, output):
    train_corpus, train_error_index = construct_corpus_from_file_vocab(train_path, vocab)
    test_corpus, test_error_index = construct_corpus_from_file_vocab(test_path, vocab)
    dump_pickle(train_corpus, os.path.join(output, 'train.corpus'))
    dump_pickle(train_error_index, os.path.join(output, 'train_error.index'))
    dump_pickle(test_corpus, os.path.join(output, 'test.corpus'))
    dump_pickle(test_error_index, os.path.join(output, 'test_error.index'))
    return train_corpus, test_corpus

def generate_labels_from_file_and_error(file_name, error_file, output):
    labels = []
    error_index = load_pickle(error_file)
    try:
        fp = open(file_name, 'r')
        _ = fp.readline()
        line_index = 0
        while True:
            line = fp.readline()
            line_index += 1
            if not line:
                break
            if line_index in error_index:
                continue
            labels_str = line.split(' ', 1)[0]
            labels_str = labels_str.split(',')
            labels_doc = [int(label) for label in labels_str]
            labels.append(labels_doc)
    except Exception as e:
        raise e
    # delete error index
    # for ind in error_index:
    #     del labels[ind]
    print 'num of y_label: ' + str(len(labels))
    dump_pickle(labels, output)
    return labels

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
            try:
                #seq_len, emb = generate_embedding(data[s], max_seq_len, vocab, word_embeddings)
                seq_len, emb = generate_embedding_from_vocabID(data[s], max_seq_len, word_embeddings)
            except Exception as e:
                print s
                raise e
            l_v = generate_label_vector(labels[s], num_label)
            batch_x.append(emb)
            batch_y.append(l_v)
            batch_l.append(seq_len)
        x.append(batch_x)
        y.append(batch_y)
        length.append(batch_l)
        i = i + batch_size
    return x, y, length

def generate_embedding_from_vocabID(sequence, max_seq_len, word_embeddings):
    embeddings = []
    seq_len = min(len(sequence), max_seq_len)
    for i in range(seq_len):
        try:
            #index = vocab.index(sequence[i])
            emb_str = word_embeddings[i]
            embeddings.append(gen_word_emb_from_str(emb_str))
        except Exception as e:
            raise e
    seq_len = len(embeddings)
    zero_len = int(max_seq_len - seq_len)
    embeddings = np.array(embeddings)
    #print embeddings.shape
    if zero_len > 0:
        zero_emb = np.zeros([zero_len, embeddings.shape[-1]])
        #print zero_emb.shape
        embeddings = np.concatenate((embeddings, zero_emb), axis=0)
    return seq_len, embeddings

def generate_embedding(sequence, max_seq_len, vocab, word_embeddings):
    embeddings = []
    seq_len = min(len(sequence), max_seq_len)
    for i in range(seq_len):
        try:
            index = vocab.index(sequence[i])
            emb_str = word_embeddings[index]
            embeddings.append(gen_word_emb_from_str(emb_str))
        except Exception:
            continue
    seq_len = len(embeddings)
    zero_len = int(max_seq_len - seq_len)
    embeddings = np.array(embeddings)
    #print embeddings.shape
    if zero_len > 0:
        zero_emb = np.zeros([zero_len, embeddings.shape[-1]])
        #print zero_emb.shape
        embeddings = np.concatenate((embeddings, zero_emb), axis=0)
    return seq_len, embeddings

def gen_word_emb_from_str(str):
    _, v_str = str.split(' ', 1)
    v = [float(e) for e in v_str.split()]
    return v

def generate_label_vector(labels, num_label):
    return np.sum(np.eye(num_label)[labels], axis=0)


