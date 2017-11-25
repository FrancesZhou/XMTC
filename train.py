'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
from biLSTM.preprocessing.preprocessing import generate_label_embedding_from_file, get_train_test_doc_label_data
from biLSTM.preprocessing.dataloader import DataLoader2, DataLoader3
from biLSTM.core.biLSTM import biLSTM
from biLSTM.core.solver import ModelSolver
from biLSTM.utils.io_utils import load_pickle, load_txt

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-doc_wordID_data', '--doc_wordID_data_path', type=str,
                       default='datasets/AmazonCat-13K/output/descriptions/label_pair/doc_wordID_data.pkl',
                       help='path to the documents')
    parse.add_argument('-label_data', '--label_data_path', type=str,
                       default='datasets/AmazonCat-13K/output/descriptions/label_pair/label_data.pkl',
                       help='path to the labels')
    parse.add_argument('-train_pid', '--train_pid_path', type=str,
                       default='datasets/AmazonCat-13K/output/descriptions/label_pair/train_pid.pkl',
                       help='path to the training pids')
    parse.add_argument('-test_pid', '--test_pid_path', type=str,
                       default='datasets/AmazonCat-13K/output/descriptions/label_pair/test_pid.pkl',
                       help='path to the testing pids')
    parse.add_argument('-vocab', '--vocab_path', type=str, default='datasets/vocab', help='path to the vocab')
    # parse.add_argument('-word_embeddings', '--word_embedding_path', type=str,
    #                    default='datasets/glove.840B.300d.txt',
    #                    help='path to the word embeddings')
    parse.add_argument('-word_embeddings', '--word_embedding_path', type=str,
                       default='datasets/word_embeddings.npy',
                       help='path to the word embeddings')
    parse.add_argument('-label_embeddings', '--label_embedding_path', type=str,
                       default='datasets/AmazonCat-13K/output/descriptions/label_pair/label.embeddings',
                       help='path to the label embeddings')
    parse.add_argument('-pretrained_model', '--pretrained_model_path', type=str,
                       default=None, help='path to the pretrained model')
    #parse.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    # -- default
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=10, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=16, help='batch size')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    args = parse.parse_args()

    # x,y,l- input
    vocab = load_pickle(args.vocab_path)
    print 'load word/label embeddings'
    # word_embeddings: readlines() from .txt file
    # word_embeddings: 'word': word_embedding\n
    #word_embeddings = load_txt(args.word_embedding_path)
    word_embeddings = np.load(args.word_embedding_path)
    #word_embeddings = []
    #label_embeddings = load_pickle(args.label_embedding_path)
    label_embeddings = generate_label_embedding_from_file(args.label_embedding_path)
    print 'load train/test data'
    doc_wordID_data = load_pickle(args.doc_wordID_data_path)
    label_data = load_pickle(args.label_data_path)
    train_pid = load_pickle(args.train_pid_path)
    test_pid = load_pickle(args.test_pid_path)
    train_doc, train_label, test_doc, test_label = get_train_test_doc_label_data(doc_wordID_data, label_data, train_pid, test_pid)
    # train_data = train_data[:len(train_data)/10]
    # test_data = test_data[:len(test_data)/10]
    # train_label = train_label[:len(train_data)]
    # test_label = test_label[:len(test_data)]
    #all_labels = load_pickle(args.all_labels_path)
    all_labels = label_embeddings.keys()
    print 'create train/test data loader...'
    train_loader = DataLoader2(train_doc, train_label, all_labels, label_embeddings, args.batch_size, vocab, word_embeddings, pos_neg_ratio=1)
    max_seq_len = train_loader.max_seq_len
    test_loader = DataLoader2(test_doc, test_label, all_labels, label_embeddings, args.batch_size, vocab, word_embeddings, pos_neg_ratio=1, max_seq_len=max_seq_len)
    # ----- train -----
    print 'build biLSTM model...'
    # (self, max_seq_len, input_dim, num_label_embedding, num_hidden, num_classify_hidden)
    model = biLSTM(max_seq_len, 300, 64, 64, 32, args.batch_size)

    print 'model solver...'
    # def __init__(self, model, train_data, test_data, **kwargs):
    solver = ModelSolver(model, train_loader, test_loader,
                         n_epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         update_rule=args.update_rule,
                         learning_rate=args.learning_rate,
                         pretrained_model=None,
                         model_path='datasets/AmazonCat-13K/output/results/model_save/',
                         test_path='datasets/AmazonCat-13K/output/results/model_save/')
    print 'begin training...'
    solver.train()

    # test
    test_all = DataLoader3(test_doc, test_label, all_labels, label_embeddings, args.batch_size, vocab, word_embeddings, max_seq_len)
    solver.test(test_all)


if __name__ == "__main__":
    main()
