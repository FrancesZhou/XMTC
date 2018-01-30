'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

#import argparse
import numpy as np
import sys
sys.path.append('../material')
from utils import load_pickle, dump_pickle, load_txt

source_path = 'data/deeplearning_data/adjacent_labels/'
des_path = 'data/baseline_data/adjacent_labels/'
train_asin_label_file = source_path + 'train_label.pkl'
test_asin_label_file = source_path + 'test_label.pkl'
train_asin_feature_file = source_path + 'train_feature.pkl'
test_asin_feature_file = source_path + 'test_feature.pkl'

train_asin_label = load_pickle(train_asin_label_file)
test_asin_label = load_pickle(test_asin_label_file)
train_asin_feature = load_pickle(train_asin_feature_file)
test_asin_feature = load_pickle(test_asin_feature_file)

train_label_feature_file = open(des_path + 'train_data.txt', 'w')
test_label_feature_file = open(des_path + 'test_data.txt', 'w')
train_map_file = open(des_path + 'train_map.txt', 'w')
test_map_file = open(des_path + 'test_map.txt', 'w')

labels = np.unique(np.concatenate(train_asin_label.values())).tolist()
num_labels = len(labels)
num_features = 203882

label_index = range(num_labels)
label_dict = dict(zip(labels, label_index))
dump_pickle(labels, des_path + 'all_labels.pkl')
dump_pickle(label_dict, des_path + 'label_dict.pkl')

# train
train_pids = train_asin_label.keys()
num_train = len(train_pids)
train_label_feature_file.write(str(num_train) + ' ' + str(num_features) + ' ' + str(num_labels) + '\n')

for pid in train_pids:
    train_map_file.write(str(pid) + '\n')
    l = train_asin_label[pid]
    l_str = ''
    for l_i in l[:-1]:
        l_str = l_str + str(label_dict[l_i]) + ','
    l_str = l_str + str(label_dict[l[-1]]) + ' '
    # add features
    l_str = l_str + train_asin_feature[pid]
    train_label_feature_file.write(l_str)

test_pids = test_asin_label.keys()
num_test = len(test_pids)
test_label_feature_file.write(str(num_test) + ' ' + str(num_features) + ' ' + str(num_labels) + '\n')
for pid in test_pids:
    test_map_file.write(str(pid) + '\n')
    l = test_asin_label[pid]
    l_str = ''
    for l_i in l[:-1]:
        l_str = l_str + str(label_dict[l_i]) + ','
    l_str = l_str + str(label_dict[l[-1]]) + ' '
    # add features
    l_str = l_str + test_asin_feature[pid]
    test_label_feature_file.write(l_str)

train_label_feature_file.close()
test_label_feature_file.close()
train_map_file.close()
test_map_file.close()