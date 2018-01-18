'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import argparse
import operator
import numpy as np
import sys
import scipy.io as sio
sys.path.append('../material')
from utils import load_pickle, dump_pickle, load_txt


def get_k_top_labels(label_score_str, k):
    label_value_pair = label_score_str.strip().split(' ')
    lv_dict = {}
    for pair in label_value_pair:
        l, v = pair.split(':')
        lv_dict[int(l)] = float(v)
    sorted_lv_dict = sorted(lv_dict.items(), key=operator.itemgetter(1), reverse=True)[:min(len(label_value_pair), k)]
    top_lv_dict = dict(sorted_lv_dict)
    return top_lv_dict.keys()

def get_candidate_labels(path, out_path, type):
    train_titles_file = path + 'train_map.txt'
    test_titles_file = path + 'test_map.txt'
    train_candidate_label_file = path + type + '_candidate/train_score_mat.txt'
    test_candidate_label_file = path + type + '_candidate/test_score_mat.txt'
    #label_index_file = path + 'label_dict.pkl'
    index_label_file = path + 'all_labels.pkl'
    train_titles = load_txt(train_titles_file)
    test_titles = load_txt(test_titles_file)
    train_candidate_all = load_txt(train_candidate_label_file)[1:]
    test_candidate_all = load_txt(test_candidate_label_file)[1:]

    index_label = load_pickle(index_label_file)
    train_candidate_labels = {}
    test_candidate_labels = {}
    # train data
    for i in range(len(train_titles)):
        pid = train_titles[i].strip()
        pid = int(pid)
        candidate_label_index = get_k_top_labels(train_candidate_all[i], 30)
        candidate_labels = [index_label[ind] for ind in candidate_label_index]
        train_candidate_labels[pid] = candidate_labels
    for i in range(len(test_titles)):
        pid = test_titles[i].strip()
        pid = int(pid)
        candidate_label_index = get_k_top_labels(test_candidate_all[i], 30)
        candidate_labels = [index_label[ind] for ind in candidate_label_index]
        test_candidate_labels[pid] = candidate_labels
    dump_pickle(train_candidate_labels, out_path + type + '_candidate/train_candidate_label.pkl')
    dump_pickle(test_candidate_labels, out_path + type + '_candidate/test_candidate_label.pkl')

def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-type', '--candidate_type', type=str,
                       default='fastxml', help='sleec or fastxml (which method used to generate candidate labels)')

    args = parse.parse_args()

    path = 'data/baseline_data/adjacent_labels/'
    out_path = 'data/deeplearning_data/adjacent_labels/'


    get_candidate_labels(path, out_path, args.candidate_type)

if __name__ == "__main__":
    main()