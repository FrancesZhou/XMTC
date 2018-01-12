'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

#import argparse
import numpy as np
import sys
import scipy.io as sio
sys.path.append('../material')
from utils import load_pickle, dump_pickle, load_txt

path = 'data/baseline_data/adjacent_labels/all_para/'
out_path = 'data/deeplearning_data/adjacent_labels/all_para/'

train_titles_file = path + 'train_map.txt'
test_titles_file = path + 'test_map.txt'
train_candidate_label_file = path + 'candidate_train.mat'
test_candidate_label_file = path + 'candidate_test.mat'
label_index_file = path + 'label_dict.pkl'


def get_candidate_labels():
    train_titles = load_txt(train_titles_file)
    test_titles = load_txt(test_titles_file)
    train_candidate_all = sio.loadmat(train_candidate_label_file)['candidate_train']
    test_candidate_all = sio.loadmat(test_candidate_label_file)['candidate_test']
    label_index = load_pickle(label_index_file)
    index_label = dict(zip(label_index.values(), label_index.keys()))
    train_candidate_labels = {}
    test_candidate_labels = {}
    # train data
    for i in range(len(train_titles)):
        pid = train_titles[i].strip()
        pid = int(pid)
        candidate_label_index = train_candidate_all[i]
        candidate_labels = [index_label[ind] for ind in candidate_label_index]
        train_candidate_labels[pid] = candidate_labels
    for i in range(len(test_titles)):
        pid = test_titles[i].strip()
        pid = int(pid)
        candidate_label_index = test_candidate_all[i]
        candidate_labels = [index_label[ind] for ind in candidate_label_index]
        test_candidate_labels[pid] = candidate_labels
    dump_pickle(train_candidate_labels, out_path + 'train_candidate_label.pkl')
    dump_pickle(test_candidate_labels, out_path + 'test_candidate_label.pkl')

get_candidate_labels()