'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
#import argparse
import json
import numpy as np
import scipy.io as sio
from model.preprocessing.preprocessing import get_wordID_from_vocab, write_label_pairs_into_file
from model.utils.io_utils import load_pickle, dump_pickle, load_txt

train_data_file = 'datasets/Wiki10/output/train_data.pkl'
test_data_file = 'datasets/Wiki10/output/test_data.pkl'
train_titles_file = 'datasets/Wiki10/output/train_titles.pkl'
test_titles_file = 'datasets/Wiki10/output/test_titles.pkl'
train_file = 'datasets/Wiki10/output/wiki10_train.txt'
test_file = 'datasets/Wiki10/output/wiki10_test.txt'

train_label_file = 'datasets/Wiki10/output/train_label.pkl'
test_label_file = 'datasets/Wiki10/output/test_label.pkl'

train_data_final = 'datasets/Wiki10/output/label_pair/train_data.pkl'
test_data_final = 'datasets/Wiki10/output/label_pair/test_data.pkl'

train_doc_wordID_final = 'datasets/Wiki10/output/final/train_doc_wordID.pkl'
test_doc_wordID_final = 'datasets/Wiki10/output/final/test_doc_wordID.pkl'
train_label_final = 'datasets/Wiki10/output/final/train_label.pkl'
test_label_final = 'datasets/Wiki10/output/final/test_label.pkl'
label_pairs_final = 'datasets/Wiki10/output/label_pair/labels_pair.pkl'
all_labels_final = 'datasets/Wiki10/output/label_pair/all_labels.pkl'

train_candidate_label_file = 'datasets/Wiki10/output/candidate_train.mat'
test_candidate_label_file = 'datasets/Wiki10/output/candidate_test.mat'
train_candidate_label_final = 'datasets/Wiki10/output/final/train_candidate_label.pkl'
test_candidate_label_final = 'datasets/Wiki10/output/final/test_candidate_label.pkl'

# generate train/test label data from train_data, test_data, train_titles, test_titles
def get_label_data():
    train_data = load_pickle(train_data_file)
    test_data = load_pickle(test_data_file)
    train_titles = load_pickle(train_titles_file)
    test_titles = load_pickle(test_titles_file)

    train_label_txt = load_txt(train_file)[1:]
    test_label_txt = load_txt(test_file)[1:]

    all_labels = []
    train_label = {}
    test_label = {}
    for title, _ in train_data.items():
        line = train_label_txt[train_titles.index(title)]
        labels_str = line.split(' ', 1)[0]
        labels_str = labels_str.split(',')
        labels = [int(label) for label in labels_str]
        all_labels.append(labels)
        train_label[title] = labels

    for title, _ in test_data.items():
        line = test_label_txt[test_titles.index(title)]
        labels_str = line.split(' ', 1)[0]
        labels_str = labels_str.split(',')
        labels = [int(label) for label in labels_str]
        test_label[title] = labels

    dump_pickle(train_label, train_label_file)
    dump_pickle(test_label, test_label_file)

# generate label pair
def generate_label_pair():
    train_label = load_pickle(train_label_file)
    all_labels = []
    # get label pairs
    label_pairs = []
    for _, labels_doc in train_label.items():
        all_labels.append(labels_doc)
        if len(labels_doc) == 1:
            continue
        labels_doc = sorted(labels_doc)
        label_pair_start = labels_doc[0]
        for label in labels_doc[1:]:
            label_pairs.append([label_pair_start, label])
    # delete duplica
    label_pairs = np.array(label_pairs, dtype=np.int32)
    label_pairs = np.unique(label_pairs, axis=0)

    all_labels = np.concatenate(all_labels)
    all_labels = np.unique(all_labels)
    all_label_pair = np.unique(np.concatenate(label_pairs))
    separate_labels = list(set(all_labels) - set(all_label_pair))
    print len(separate_labels)
    dump_pickle(label_pairs, label_pairs_final)
    dump_pickle(all_label_pair, all_labels_final)
    return all_label_pair, separate_labels

def get_valid_train_test_data():
    train_label = load_pickle(train_label_file)
    test_label = load_pickle(test_label_file)
    train_data = load_pickle(train_data_file)
    test_data = load_pickle(test_data_file)
    all_labels = load_pickle(all_labels_final)
    missing_labels = np.array(list(set(range(30938)) - set(all_labels)))
    for pid, l in train_label.items():
        l2 = list(set(l) - set(missing_labels))
        if len(l2)>0:
            train_label[pid] = l2
        else:
            del train_label[pid]
            del train_data[pid]
    for pid, l in test_label.items():
        l2 = list(set(l) - set(missing_labels))
        if len(l2):
            test_label[pid] = l2
        else:
            del test_label[pid]
            del test_data[pid]
    dump_pickle(train_label, train_label_final)
    dump_pickle(train_data, train_data_final)
    dump_pickle(test_label, test_label_final)
    dump_pickle(test_data, test_data_final)

def get_train_test_doc_data():
    vocab = load_pickle('datasets/vocab')
    train_data = load_pickle(train_data_final)
    test_data = load_pickle(test_data_final)
    train_doc_wordID = {}
    test_doc_wordID = {}

    for id, text in train_data.items():
        text = id + '. ' + text
        wordID = get_wordID_from_vocab(text, vocab)
        train_doc_wordID[id] = wordID
    for id, text in test_data.items():
        text = id + '. ' + text
        wordID = get_wordID_from_vocab(text, vocab)
        test_doc_wordID[id] = wordID
    dump_pickle(train_doc_wordID, train_doc_wordID_final)
    dump_pickle(test_doc_wordID, test_doc_wordID_final)

def get_number_of_all_positive_samples():
    train_label = load_pickle(train_label_final)
    test_label = load_pickle(test_label_final)
    train_all_pos = np.concatenate(train_label.values())
    test_all_pos = np.concatenate(test_label.values())
    print len(train_all_pos)
    print len(test_all_pos)

# get candidate labels from SLEEC results
def get_candidate_labels():
    train_titles = load_pickle(train_titles_file)
    test_titles = load_pickle(test_titles_file)
    train_candidate_all = sio.loadmat(train_candidate_label_file)['candidate_train']
    test_candidate_all = sio.loadmat(test_candidate_label_file)['candidate_test']
    train_label = load_pickle(train_label_final)
    test_label = load_pickle(test_label_final)
    train_candidate_labels = {}
    test_candidate_labels = {}
    # train data
    for pid, _ in train_label.items():
        candidate_labels = train_candidate_all[train_titles.index(pid)]
        train_candidate_labels[pid] = candidate_labels.tolist()
    for pid, _ in test_label.items():
        candidate_labels = test_candidate_all[test_titles.index(pid)]
        test_candidate_labels[pid] = candidate_labels.tolist()
    dump_pickle(train_candidate_labels, train_candidate_label_final)
    dump_pickle(test_candidate_labels, test_candidate_label_final)

#get_label_data()
#generate_label_pair()
#get_valid_train_test_data()

# get_train_test_doc_data()

get_candidate_labels()

# write_label_pairs_into_file(label_pairs_final, 'datasets/Wiki10/output/label_pair/labels.edgelist')
#get_number_of_all_positive_samples()

