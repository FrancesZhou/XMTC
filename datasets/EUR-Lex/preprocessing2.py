'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import argparse
import numpy as np
import sys
import  math
from collections import Counter
sys.path.append('../material')
#from ..material.utils import load_pickle, dump_pickle, load_txt, get_wordID_from_vocab
from utils import load_pickle, dump_pickle, load_txt, get_wordID_from_vocab_dict_for_raw_text, write_label_pairs_into_file, get_titles_from_map_file

data_des_path = 'trn_tst_data/'

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
    dump_pickle(train_doc_wordID, data_des_path + 'train_doc_wordID.pkl')
    dump_pickle(train_feature_str, data_des_path + 'train_feature.pkl')
    dump_pickle(train_label, data_des_path + 'train_label.pkl')
    dump_pickle(test_doc_wordID, data_des_path + 'test_doc_wordID.pkl')
    dump_pickle(test_feature_str, data_des_path + 'test_feature.pkl')
    dump_pickle(test_label, data_des_path + 'test_label.pkl')
    return all_labels, train_doc_wordID, train_label, test_doc_wordID, test_label, train_feature_str, test_feature_str


# Wikipedia-LSHTC: A=0.5,  B=0.4
# Amazon:          A=0.6,  B=2.6
# Other:		   A=0.55, B=1.5
def get_label_propensity(train_pid_label, A=0.55, B=1.5):
    inv_prop_file = data_des_path + 'inv_prop.txt'
    train_label = train_pid_label.values()
    train_label = np.concatenate(train_label).tolist()
    label_frequency = dict(Counter(train_label))
    labels, fre = zip(*label_frequency.iteritems())
    fre = np.array(fre)

    N = len(train_pid_label)
    C = (math.log(N)-1) * (B + 1)**A
    inv_prop = 1 + C * (fre + B)**(-A)

    inv_prop_dict = dict(zip(labels, inv_prop.tolist()))
    dump_pickle(inv_prop_dict, data_des_path + 'inv_prop_dict.pkl')
    #
    with open(inv_prop_file, 'w') as df:
        for l_, prop_ in inv_prop_dict.items():
            df.write(str(l_) + ': ' + str(prop_))
            df.write('\n')

def main():
    train_label_fea_file = 'sources/xml/eurlex_train.txt'
    test_label_fea_file = 'sources/xml/eurlex_test.txt'
    all_labels, train_doc_wordID, train_label, test_doc_wordID, test_label, train_feature, test_feature = \
            get_train_test_data(train_label_fea_file, test_label_fea_file)
    get_label_propensity(train_label, 0.55, 1.5)

if __name__ == "__main__":
    main()