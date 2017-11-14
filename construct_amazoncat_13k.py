'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
from biLSTM.preprocessing.preprocessing import construct_train_test_corpus, generate_labels_from_file, generate_label_pair_from_file
from biLSTM.utils.io_utils import load_pickle, write_file

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-train_corpus', '--train_corpus_path', type=str, required=True, help='path to the training corpus')
    parse.add_argument('-test_corpus', '--test_corpus_path', type=str, required=True, help='path to the testing corpus')
    parse.add_argument('-train_labels', '--train_labels_path', type=str, required=True, help='path to the training labels')
    parse.add_argument('-test_labels', '--test_labels_path', type=str, required=True, help='path to the testing labels')
    parse.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    parse.add_argument('-if_label_pair', '--if_label_pair', type=bool, required=True, help='whether to generate label pairs')
    args = parse.parse_args()

    # train_corpus: 'datasets/AmazonCat-13K/AmazonCat-13K_train_map.txt'
    # test_corpus: 'datasets/AmazonCat-13K/AmazonCat-13K_test_map.txt'
    # train_labels: 'datasets/AmazonCat-13K/amazonCat_train.txt'
    # test_labels: 'datasets/AmazonCat-13K/amazonCat_test.txt'
    # out_dir: 'datasets/AmazonCat-13K/output/'
    ## ----------- get train/test corpus -----------
    train_corpus, test_corpus = construct_train_test_corpus(args.train_corpus_path, args.test_corpus_path, args.out_dir)
    ## ----------- get train/test labels -----------
    # train_labels = generate_labels_from_file(args.train_labels_path, os.path.join(args.out_dir, 'train.labels'))
    # test_labels = generate_labels_from_file(args.test_labels_path, os.path.join(args.out_dir, 'test.labels'))
    # ----------- generate label pairs (used by deepwalk) ----------
    if args.if_label_pair:
        label_pairs = generate_label_pair_from_file(os.path.join(args.out_dir, 'train.labels'), os.path.join(args.out_dir, 'labels.pair'))
        # ------------ analysis of labels ------------
        # label_pairs = load_pickle(os.path.join(args.out_dir, 'labels.pair'))
        # l = label_pairs.flatten()
        # l = np.unique(l)
        # print len(l)
        # print max(l)
        # print min(l)
        # # find those separate labels
        # all_labels = load_pickle(os.path.join(args.out_dir, 'train.labels'))
        # all_labels = np.hstack(all_labels)
        # all_labels = np.unique(all_labels)
        # print len(all_labels)
        # print max(all_labels)
        # print min(all_labels)
        # print(set(all_labels)-set(l))
        # print 'done.'




if __name__ == "__main__":
    main()