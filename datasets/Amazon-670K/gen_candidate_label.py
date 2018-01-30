'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import argparse
import numpy as np
import sys
import scipy.io as sio
sys.path.append('../material')
from utils import load_pickle, dump_pickle, load_txt


def get_candidate_labels(path, out_path, type, format):
    train_titles_file = path + 'train_map.txt'
    test_titles_file = path + 'test_map.txt'
    # train_candidate_label_file = path + type + '_candidate/candidate_train.mat'
    # test_candidate_label_file = path + type + '_candidate/candidate_test.mat'
    #label_index_file = path + 'label_dict.pkl'
    index_label_file = path + 'all_labels.pkl'
    train_titles = load_txt(train_titles_file)
    test_titles = load_txt(test_titles_file)
    index_label = load_pickle(index_label_file)
    train_candidate_labels = {}
    test_candidate_labels = {}
    if format == 'mat':
        train_candidate_label_file = path + type + '_candidate/candidate_train.mat'
        test_candidate_label_file = path + type + '_candidate/candidate_test.mat'
        train_candidate_all = sio.loadmat(train_candidate_label_file)['candidate_train']
        test_candidate_all = sio.loadmat(test_candidate_label_file)['candidate_test']
        for i in xrange(len(train_titles)):
            pid = train_titles[i].strip()
            pid = int(pid)
            candidate_label_index = train_candidate_all[i]
            candidate_labels = [index_label[ind] for ind in candidate_label_index]
            train_candidate_labels[pid] = candidate_labels
        for i in xrange(len(test_titles)):
            pid = test_titles[i].strip()
            pid = int(pid)
            candidate_label_index = test_candidate_all[i]
            candidate_labels = [index_label[ind] for ind in candidate_label_index]
            test_candidate_labels[pid] = candidate_labels
    elif format == 'txt':
        train_candidate_label_file = path + type + '_candidate/train_score_mat.txt'
        test_candidate_label_file = path + type + '_candidate/test_score_mat.txt'
        train_candidate_all = load_txt(train_candidate_label_file)[1:]
        test_candidate_all = load_txt(test_candidate_label_file)[1:]
        for i in xrange(len(train_titles)):
            pid = int(train_titles[i].strip())
            candidate_label_line = train_candidate_all[i].strip()
            candidate_label_score = {}
            for l_s in candidate_label_line.split(' ')[:50]:
                l_, s_ = l_s.split(':')
                ll = index_label[int(l_)]
                candidate_label_score[ll] = float(s_)
            train_candidate_labels[pid] = candidate_label_score
        for i in xrange(len(test_titles)):
            pid = int(test_titles[i].strip())
            candidate_label_line = test_candidate_all[i].strip()
            candidate_label_score = {}
            for l_s in candidate_label_line.split(' ')[:50]:
                l_, s_ = l_s.split(':')
                ll = index_label[int(l_)]
                candidate_label_score[ll] = float(s_)
            test_candidate_labels[pid] = candidate_label_score

    dump_pickle(train_candidate_labels, out_path + type + '_candidate/train_candidate_label.pkl')
    dump_pickle(test_candidate_labels, out_path + type + '_candidate/test_candidate_label.pkl')

def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-data_format', '--data_format', type=str,
                       default='txt', help='mat file or txt file (which way to store candidate labels)')
    parse.add_argument('-can_type', '--candidate_type', type=str,
                       default='sleec', help='sleec or fastxml (which method to generate candidate labels)')

    args = parse.parse_args()

    path = 'data/baseline_data/adjacent_labels/'
    out_path = 'data/deeplearning_data/adjacent_labels/'

    get_candidate_labels(path, out_path, args.candidate_type, args.data_format)

if __name__ == "__main__":
    main()