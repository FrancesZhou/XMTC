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

# input: a text with several words, vocab
# output: a list consisted of wordIDs in vocab
def get_wordID_from_vocab(text, vocab):
    all_tokens = re.split('([A-Za-z]+)|([0-9]+)|(\W)', text)
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
    return token_indices

# input: corpus_file (N titles after pid->)
# output:
# N lists which contain wordIDs in vocab,
# error_index which refers to those invalid title words
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
    # print len(corpus)
    # print len(error_index)
    return corpus, error_index

# construct train/test corpus
def construct_train_test_corpus(vocab, train_path, test_path, output):
    train_corpus, train_error_index = construct_corpus_from_file_vocab(train_path, vocab)
    test_corpus, test_error_index = construct_corpus_from_file_vocab(test_path, vocab)
    dump_pickle(train_corpus, os.path.join(output, 'train.corpus'))
    dump_pickle(train_error_index, os.path.join(output, 'train_error.index'))
    dump_pickle(test_corpus, os.path.join(output, 'test.corpus'))
    dump_pickle(test_error_index, os.path.join(output, 'test_error.index'))
    return train_corpus, test_corpus

# input: label_file (N label+feature)
# output: labels which contain positive labels for each line
# delete error lines
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
    print 'num of y_label: ' + str(len(labels))
    dump_pickle(labels, output)
    return labels

# input: label_file (N label+feature)
# output: labels which contain positive labels for each product
def generate_labels_from_file(file_name, output):
    labels = []
    try:
        fp = open(file_name, 'r')
        _ = fp.readline()
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

# input: labels (lists) which contain positive labels for each product
# output: label pairs used for creating label.edgelist
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

# input: data
# output: get maximum length of all data
def get_max_seq_len(data):
    num = len(data)
    all_seq_len = np.zeros(num)
    for i in range(num):
        all_seq_len[i] = len(data[i])
    max_seq_len = max(all_seq_len)
    return max_seq_len

# input: labels (lists)
# output: get maximum/mean number of positive labels in all products
def get_max_num_labels(labels):
    num = len(labels)
    all_num_labels = np.zeros(num)
    for i in range(num):
        all_num_labels[i] = len(labels[i])
    max_num_labels = max(all_num_labels)
    mean_num_labels = np.mean(all_num_labels)
    return max_num_labels, mean_num_labels

def batch_data(data, labels, max_seq_len, num_labels, vocab, word_embeddings, batch_size=32):
    num = len(data)
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
                seq_len, emb = generate_embedding_from_vocabID(data[s], max_seq_len, word_embeddings)
            except Exception as e:
                print s
                raise e
            l_v = generate_label_vector(labels[s], num_labels)
            batch_x.append(emb)
            batch_y.append(l_v)
            batch_l.append(seq_len)
        x.append(batch_x)
        y.append(batch_y)
        length.append(batch_l)
        i = i + batch_size
    return x, y, length

# input: sequence which contains word indices in vocab, maximum sequence length, word_embeddings
# output: real sequence length, embedding matrix for sequence
def generate_embedding_from_vocabID(sequence, max_seq_len, word_embeddings):
    embeddings = []
    seq_len = min(len(sequence), max_seq_len)
    for i in range(seq_len):
        try:
            embeddings.append(word_embeddings[sequence[i]])
            # emb_str = word_embeddings[sequence[i]]
            # embeddings.append(gen_word_emb_from_str(emb_str))
        except Exception as e:
            raise e
    seq_len = len(embeddings)
    zero_len = int(max_seq_len - seq_len)
    embeddings = np.array(embeddings)
    #print embeddings.shape
    if zero_len > 0:
        zero_emb = np.zeros((zero_len, embeddings.shape[-1]))
        #print zero_emb.shape
        print zero_emb.shape
        print embeddings.shape
        embeddings = np.concatenate((embeddings, zero_emb), axis=0)
    return seq_len, embeddings

# input: sequence which contains words, maximum sequence length, vocab, word_embeddings
# output: real sequence length, embedding matrix for sequence
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

# input: line in word_embedding txt
# output: word embedding
def gen_word_emb_from_str(str):
    _, v_str = str.split(' ', 1)
    v = [float(e) for e in v_str.split()]
    return v

def generate_label_vector(labels, num_labels):
    return np.sum(np.eye(num_labels)[labels], axis=0)

def generate_label_vector_of_fixed_length(pos_labels, num_labels, num_all_labels):
    all_labels = np.arange(num_all_labels)
    all_neg_labels = np.array(list(set(all_labels) - set(pos_labels)))
    neg_labels = np.random.choice(all_neg_labels, num_labels-len(pos_labels))
    index_labels = np.concatenate([pos_labels, neg_labels])
    value_labels = np.concatenate([np.ones(len(pos_labels), dtype=int), np.zeros(len(neg_labels), dtype=int)])
    zipped = zip(index_labels, value_labels)
    zipped = sorted(zipped, key=lambda x: x[0])
    label_indices, labels = zip(*zipped)
    return label_indices, labels

# input: doc_data, label_data, train_pid, test_pid
# output: train_doc, train_label, test_doc, test_label
def get_train_test_doc_label_data(doc_data, label_data, train_pid, test_pid):
    train_doc = {}
    train_label = {}
    for pid in train_pid:
        train_doc[pid] = doc_data[pid]
        train_label[pid] = label_data[pid]
    test_doc = {}
    test_label = {}
    for pid in test_pid:
        test_doc[pid] = doc_data[pid]
        test_label[pid] = label_data[pid]
    return train_doc, train_label, test_doc, test_label

# input: label_embedding txt
# output: dict of label_embedding: {label: embedding, ...}
def generate_label_embedding_from_file(file):
    label_embeddings = {}
    with open(file, 'r') as df:
        lines = df.readlines()
        for line in lines[1:]:
            label, v_str = line.split(' ', 1)
            v = [float(e) for e in v_str.split()]
            label_embeddings[int(label)] = v
    return label_embeddings

def write_label_pairs_into_file(label_pairs_file, output_file):
    txtfile = open(output_file, 'w')
    label_pairs = load_pickle(label_pairs_file)
    for i in range(len(label_pairs)):
        txtfile.write(str(label_pairs[i][0]) + '\t' + str(label_pairs[i][1]))
        txtfile.write('\n')

