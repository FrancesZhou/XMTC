'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
from biLSTM.preprocessing.preprocessing import get_max_seq_len, construct_train_test_corpus, generate_labels_from_file, generate_label_pair_from_file
from biLSTM.preprocessing.dataloader import DataLoader
from biLSTM.core.biLSTM import biLSTM
from biLSTM.core.solver import ModelSolver
from biLSTM.utils.io_utils import load_pickle, write_file, load_txt

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-train_corpus', '--train_corpus_path', type=str, default='datasets/AmazonCat-13K/output/train.corpus', help='path to the training corpus')
    parse.add_argument('-test_corpus', '--test_corpus_path', type=str, default='datasets/AmazonCat-13K/output/test.corpus', help='path to the testing corpus')
    parse.add_argument('-train_labels', '--train_labels_path', type=str, default='datasets/AmazonCat-13K/output/train.labels', help='path to the training labels')
    parse.add_argument('-test_labels', '--test_labels_path', type=str, default='datasets/AmazonCat-13K/output/test.labels', help='path to the testing labels')

    parse.add_argument('-vocab', '--vocab_path', type=str, default='datasets/vocab', help='path to the testing labels')
    parse.add_argument('-word_embeddings', '--word_embedding_path', type=str, default='datasets/glove.840B.300d.txt', help='path to the word embeddings')
    parse.add_argument('-label_embeddings', '--label_embedding_path', type=str, default='datasets/AmazonCat-13K/all_labels.embeddings',
                       help='path to the label embeddings')
    #parse.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    # -- default
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=30, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=16, help='batch size')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    args = parse.parse_args()

    # x,y,l- input
    #vocab = load_pickle(args.vocab_path)
    print 'load word/label embeddings'
    word_embeddings = load_txt(args.word_embedding_path)
    label_embeddings = load_pickle(args.label_embedding_path)
    num_labels = len(label_embeddings)
    print 'load train/test data'
    train_data = load_pickle(args.train_corpus_path)
    test_data = load_pickle(args.test_corpus_path)
    train_label = load_pickle(args.train_labels_path)
    test_label = load_pickle(args.test_labels_path)

    max_seq_len = get_max_seq_len(train_data)
    # batch_data(data, labels, max_seq_len, num_label, vocab, word_embeddings, batch_size=32):
    print 'create train/test data loader...'
    # train_x, train_y, train_l = batch_data(train_data, train_label, max_seq_len, num_label, vocab, word_embeddings, args.batch_size)
    # test_x, test_y, test_l = batch_data(test_data, test_label, max_seq_len, num_label, vocab, word_embeddings, args.batch_size)
    train_loader = DataLoader(train_data, train_label, args.batch_size, max_seq_len, num_labels, word_embeddings)
    test_loader = DataLoader(test_data, test_label, args.batch_size, max_seq_len, num_labels, word_embeddings)

    # ----- train -----
    # train = {'x': train_x, 'y': train_y, 'l': train_l}
    # test = {'x': test_x, 'y': test_y, 'l': test_l}
    print 'build biLSTM model...'
    # def __init__(self, seq_max_len, input_dim, num_label, num_hidden, num_classify_hidden, label_embeddings):
    model = biLSTM(max_seq_len, 300, num_labels, 512, 128, label_embeddings, args.batch_size)

    print 'model solver...'
    # def __init__(self, model, train_data, test_data, **kwargs):
    solver = ModelSolver(model, train_loader, test_loader,
                         n_epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         update_rule=args.update_rule,
                         learning_rate=args.learning_rate)
    print 'begin training...'
    solver.train()



if __name__ == "__main__":
    main()
