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


