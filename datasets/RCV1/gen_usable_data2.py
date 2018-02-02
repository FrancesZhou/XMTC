'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import argparse
import numpy as np
import re
import sys
import collections
sys.path.append('../material')
#from ..material.utils import load_pickle, dump_pickle, load_txt, get_wordID_from_vocab
from utils import load_pickle, dump_pickle, load_txt, get_wordID_from_vocab_dict_for_raw_text, write_label_pairs_into_file, get_titles_from_map_file

data_source_path = 'data/deeplearning_data/docs/xml_data/'
data_des_path = 'data/deeplearning_data/xml_data/'

def get_train_test_data(train_label_fea_file, test_label_fea_file):
    train_label_fea = load_txt(train_label_fea_file)[1:]
    test_label_fea = load_txt(test_label_fea_file)[1:]
    all_labels = []
    # train
    train_doc_wordID = {}
    train_label = {}
    train_feature_str = {}
    for i in xrange(len(train_label_fea)):
        line = train_label_fea[i]
        labels_str, feature_str = line.split(' ', 1)
        # label
        labels_str = labels_str.split(',')
        try:
            labels = [int(label) for label in labels_str]
        except ValueError:
            continue
        all_labels.append(labels)
        train_label[i] = labels
        # feature
        train_feature_str[i] = feature_str
        # word
        word_tfidf = {}
        for str in feature_str.split(' '):
            word, tfidf = str.split(':')
            word_tfidf[int(word)] = float(tfidf)
        train_doc_wordID[i] = word_tfidf
    # test
    test_doc_wordID = {}
    test_label = {}
    test_feature_str = {}
    for i in xrange(len(test_label_fea)):
        line = test_label_fea[i]
        labels_str, feature_str = line.split(' ', 1)
        # label
        labels_str = labels_str.split(',')
        try:
            labels = [int(label) for label in labels_str]
        except ValueError:
            continue
        test_label[i] = labels
        # feature
        test_feature_str[i] = feature_str
        # word
        word_tfidf = {}
        for str in feature_str.split(' '):
            word, tfidf = str.split(':')
            word_tfidf[int(word)] = float(tfidf)
        test_doc_wordID[i] = word_tfidf
    all_labels = np.unique(np.concatenate(all_labels)).tolist()
    dump_pickle(train_feature_str, data_source_path + 'train_feature.pkl')
    dump_pickle(test_feature_str, data_source_path + 'test_feature.pkl')
    dump_pickle(train_doc_wordID, data_source_path + 'train_doc_wordID.pkl')
    dump_pickle(train_label, data_source_path + 'train_label.pkl')
    dump_pickle(test_doc_wordID, data_source_path + 'test_doc_wordID.pkl')
    dump_pickle(test_label, data_source_path + 'test_label.pkl')
    return all_labels, train_doc_wordID, train_label, test_doc_wordID, test_label, train_feature_str, test_feature_str


def generate_label_pair(train_label):
    # get label pairs
    label_pairs = []
    for _, labels_doc in train_label.items():
        if len(labels_doc) == 1:
            continue
        labels_doc = sorted(labels_doc)
        label_pair_start = labels_doc[0]
        for label in labels_doc[1:]:
            label_pairs.append([label_pair_start, label])
    # delete duplica
    label_pairs = np.array(label_pairs, dtype=np.int32)
    label_pairs = np.unique(label_pairs, axis=0)
    all_adjacent_labels = np.unique(np.concatenate(label_pairs))
    return label_pairs, all_adjacent_labels

def get_valid_train_test_data(all_adjacent_labels,
                              train_doc_data, test_doc_data,
                              train_label, test_label,
                              train_feature=None, test_feature=None
                              ):
    train_pids = train_doc_data.keys()
    for pid in train_pids:
        #l = train_title_label[pid]
        l2 = list(set(train_label[pid]) & set(all_adjacent_labels))
        if len(l2):
            train_label[pid] = l2
        else:
            del train_label[pid]
            del train_doc_data[pid]
            del train_feature[pid]
    test_pids = test_doc_data.keys()
    for pid in test_pids:
        l2 = list(set(test_label[pid]) & set(all_adjacent_labels))
        if len(l2):
            test_label[pid] = l2
        else:
            del test_label[pid]
            del test_doc_data[pid]
            del test_feature[pid]
    return train_doc_data, test_doc_data, train_label, test_label, train_feature, test_feature


# get label embeddings for all_labels
# def get_label_embedding_for_all_labels(label_map_file)

def main():
    train_label_fea_file = 'sources/xml/rcv1x_train.txt'
    test_label_fea_file = 'sources/xml/rcv1x_test.txt'

    parse = argparse.ArgumentParser()

    parse.add_argument('-which_labels', '--which_labels', type=str,
                       default='adjacent',
                       help='adjacent labels or all labels')
    args = parse.parse_args()

    all_labels, train_doc_wordID, train_label, test_doc_wordID, test_label, train_feature, test_feature = \
            get_train_test_data(train_label_fea_file, test_label_fea_file)
    if args.which_labels == 'adjacent':
        # 1. label embedding
        label_pairs, all_adjacent_labels = generate_label_pair(train_label)
        print len(set(all_labels) - set(all_adjacent_labels))
        # 2. write label_pairs to txt file
        write_label_pairs_into_file(label_pairs, data_des_path + 'labels.edgelist')
        # 3. get valid train/test doc_data and title_label
        train_doc_wordID, test_doc_wordID, train_label, test_label, train_feature, test_feature = \
            get_valid_train_test_data(
                all_adjacent_labels,
                train_doc_wordID, test_doc_wordID,
                train_label, test_label,
                train_feature, test_feature
        )
        dump_pickle(train_doc_wordID, data_des_path + 'train_doc_wordID.pkl')
        dump_pickle(test_doc_wordID, data_des_path + 'test_doc_wordID.pkl')
        dump_pickle(train_label, data_des_path + 'train_label.pkl')
        dump_pickle(test_label, data_des_path + 'test_label.pkl')
        dump_pickle(train_feature, data_des_path + 'train_feature.pkl')
        dump_pickle(test_feature, data_des_path + 'test_feature.pkl')

    elif args.which_labels == 'all':
        pass
        # TODO




if __name__ == "__main__":
    main()