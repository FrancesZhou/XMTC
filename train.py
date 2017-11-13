'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
from biLSTM.preprocessing.preprocessing import batch_data, get_max_seq_len, construct_train_test_corpus, generate_labels_from_file, generate_label_pair_from_file
from biLSTM.utils.io_utils import load_pickle, write_file, load_txt

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-train_corpus', '--train_corpus_path', type=str, required=True, help='path to the training corpus')
    parse.add_argument('-test_corpus', '--test_corpus_path', type=str, required=True, help='path to the testing corpus')
    parse.add_argument('-train_labels', '--train_labels_path', type=str, required=True, help='path to the training labels')
    parse.add_argument('-test_labels', '--test_labels_path', type=str, required=True, help='path to the testing labels')
    parse.add_argument('-num_labels', '--num_labels', type=int, required=True, help='number of labels')
    parse.add_argument('-batch_size', '--batch_size', type=int, required=True, help='batch_size')
    parse.add_argument('-vocab', '--vocab_path', type=str, required=True, help='path to the testing labels')
    parse.add_argument('-word_embeddings', '--word_embedding_path', type=str, required=True, help='path to the testing labels')
    parse.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    #parse.add_argument('-if_label_pair', '--if_label_pair', type=bool, required=True, help='whether to generate label pairs')
    args = parse.parse_args()

    # x,y,l- input
    vocab = load_pickle(args.vocab_path)
    word_embeddings = load_txt(args.word_embedding_path)
    train_data = load_pickle(args.train_corpus_path)
    test_data = load_pickle(args.train_corpus_path)
    train_label = load_pickle(args.train_labels_path)
    test_label = load_pickle(args.test_labels_path)

    max_seq_len = get_max_seq_len(train_data)
    # batch_data(data, labels, max_seq_len, num_label, vocab, word_embeddings, batch_size=32):
    train_x, train_y, train_l = batch_data(train_data, train_label, max_seq_len, args.num_labels, vocab, word_embeddings, args.batch_size)
    test_x, test_y, test_l = batch_data(test_data, test_label, max_seq_len, args.num_labels, vocab, word_embeddings, args.batch_size)

    # label- embedding



if __name__ == "__main__":
    main()