'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
import scipy.io as sio
from model.preprocessing.preprocessing import generate_label_embedding_from_file, get_train_test_doc_label_data
from model.preprocessing.dataloader import DataLoader2, DataLoader4
from model.core.biLSTM import biLSTM
from model.core.LSTM import LSTM
from model.core.CNN import CNN
from model.core.solver import ModelSolver
from model.utils.io_utils import load_pickle, load_txt

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-folder', '--folder_path', type=str,
                       default='datasets/Wiki10/data/deeplearning_data/adjacent_labels/all_para/',
                       help='path to train/test data')

    parse.add_argument('-vocab', '--vocab_path', type=str, default='datasets/material/vocab', help='path to the vocab')
    parse.add_argument('-word_embeddings', '--word_embedding_path', type=str,
                       default='datasets/material/word_embeddings.npy',
                       help='path to the word embeddings')

    parse.add_argument('-pretrained_model', '--pretrained_model_path', type=str,
                       default=None, help='path to the pretrained model')
    # =============== params for CNN ==================
    parse.add_argument('-num_filters', '--num_filters', type=int,
                       default=32, help='number of filters in CNN')
    parse.add_argument('-pooling_units', '--pooling_units', type=int,
                       default=64, help='number of pooling units')
    parse.add_argument('-dropout_keep_prob', '--dropout_keep_prob', type=float,
                       default=0.5, help='keep probability in dropout layer')
    filter_sizes = [2, 4, 8]
    # -- default
    parse.add_argument('-model', '--model', type=str, default='biLSTM', help='model: LSTM, biLSTM, CNN')
    parse.add_argument('-if_all_true', '--if_all_true', type=int, default=0, help='if use all true labels for training')
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=10, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=16, help='batch size')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    args = parse.parse_args()

    print 'load vocab and word embeddings'
    vocab = load_pickle(args.vocab_path)
    word_embeddings = np.load(args.word_embedding_path)
    print 'load label embeddings'
    label_embeddings = generate_label_embedding_from_file(args.folder_path + 'label.embeddings')
    all_labels = label_embeddings.keys()
    print 'load train/test data'
    train_doc = load_pickle(args.folder_path + 'train_doc_wordID.pkl')
    test_doc = load_pickle(args.folder_path + 'test_doc_wordID.pkl')
    train_label = load_pickle(args.folder_path + 'train_title_label.pkl')
    test_label = load_pickle(args.folder_path + 'test_title_label.pkl')
    train_candidate_label = load_pickle(args.folder_path + 'train_candidate_label.pkl')
    test_candidate_label = load_pickle(args.folder_path + 'test_candidate_label.pkl')
    print 'number of labels: ' + str(len(all_labels))
    print 'create train/test data loader...'
    train_loader = DataLoader4(train_doc, train_label, train_candidate_label, all_labels, label_embeddings, args.batch_size, vocab, word_embeddings, given_seq_len=False, max_seq_len=5000)
    max_seq_len = train_loader.max_seq_len
    print 'max_seq_len: ' + str(max_seq_len)
    test_loader = DataLoader4(test_doc, test_label, test_candidate_label, all_labels, label_embeddings, args.batch_size, vocab, word_embeddings, given_seq_len=True, max_seq_len=max_seq_len)
    # ----------------------- train ------------------------
    # (self, max_seq_len, input_dim, num_label_embedding, num_hidden, num_classify_hidden)
    num_label_embedding = len(label_embeddings[0])
    print 'build model ...'
    if args.model == 'biLSTM':
        print 'build biLSTM model ...'
        model = biLSTM(max_seq_len, 300, num_label_embedding, 64, 32, args.batch_size)
    elif args.model == 'LSTM':
        print 'build LSTM model ...'
        model = LSTM(max_seq_len, 300, num_label_embedding, 64, 32, args.batch_size)
    elif args.model == 'CNN':
        print 'build CNN model ...'
        model = CNN(max_seq_len, 300, filter_sizes, num_label_embedding, 32, args)

    print 'model solver ...'
    # def __init__(self, model, train_data, test_data, **kwargs):
    solver = ModelSolver(model, train_loader, test_loader,
                         n_epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         update_rule=args.update_rule,
                         learning_rate=args.learning_rate,
                         pretrained_model=args.pretrained_model_path,
                         model_path=args.folder_path + args.model + '/',
                         test_path=args.folder_path + args.model + '/')
    print 'begin training...'
    solver.train(args.folder_path + args.model + '/outcome.txt')

    # test
    # test_all = DataLoader3(test_doc, test_label, all_labels, label_embeddings, args.batch_size, vocab, word_embeddings, max_seq_len)
    # solver.test(test_all)


if __name__ == "__main__":
    main()
