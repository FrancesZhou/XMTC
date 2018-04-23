'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
from model.preprocessing.preprocessing import generate_label_embedding_from_file_2
from model.preprocessing.dataloader import *
from model.core.NN import NN
from model.core.solver3 import ModelSolver3
from model.utils.io_utils import load_pickle
from datasets.material.utils import read_label_pairs


def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    # ---------- foler path of train/test data -------
    parse.add_argument('-folder', '--folder_path', type=str,
                       default='datasets/EUR-Lex/trn_tst_data/',
                       help='path to train/test data')
    # ---------- vocab and word embeddings --------
    parse.add_argument('-word_embedding_dim', '--word_embedding_dim', type=int, default=100, help='dim of word embedding')
    # ---------- model ----------
    parse.add_argument('-max_seq_len', '--max_seq_len', type=int, default=500, help='maximum sequence length')
    parse.add_argument('-model', '--model', type=str, default='NN', help='model: NN, LSTM, biLSTM, CNN')
    parse.add_argument('-use_attention', '--use_attention', type=int, default=1, help='whether to use attention')
    parse.add_argument('-pretrained_model', '--pretrained_model_path', type=str, default=None, help='path to the pretrained model')
    parse.add_argument('-dropout_keep_prob', '--dropout_keep_prob', type=float,
                       default=0.5, help='keep probability in dropout layer')
    # ---------- training parameters --------
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=10, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=32, help='batch size for training')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    # ------ train or predict -------
    parse.add_argument('-train', '--train', type=int, default=1, help='if training')
    parse.add_argument('-test', '--test', type=int, default=0, help='if testing')
    parse.add_argument('-predict', '--predict', type=int, default=0, help='if predicting')
    args = parse.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print '-------------- load labels ------------------------'
    label_prop_dict = load_pickle(args.folder_path + 'inv_prop_dict.pkl')
    all_labels = label_prop_dict.keys()
    num_labels = np.max(all_labels) + 1
    label_prop = np.zeros(num_labels)
    for i in xrange(num_labels):
        try:
            label_prop[i] = label_prop_dict[i]
        except KeyError:
            label_prop[i] = 1.0
    print 'real number of labels: ' + str(len(all_labels))
    print 'maximum label: ' + str(np.max(all_labels))
    print 'minimum label: ' + str(np.min(all_labels))
    print 'number of labels: ' + str(num_labels)
    print '-------------- load train/test data -------------------------'
    train_doc = load_pickle(args.folder_path + 'train_doc_wordID.pkl')
    test_doc = load_pickle(args.folder_path + 'test_doc_wordID.pkl')
    train_label = load_pickle(args.folder_path + 'train_label.pkl')
    test_label = load_pickle(args.folder_path + 'test_label.pkl')
    print '============== create train/test data loader ...'
    train_loader = DataLoader_all(train_doc, train_label, num_labels, label_prop_dict,
                                  batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    test_loader = DataLoader_all(test_doc, test_label, num_labels, label_prop_dict,
                                 batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    # ----------------------- train ------------------------
    print '============== build model ...'
    if 'NN' in args.model:
        print 'build NN model ...'
        model = NN(args.max_seq_len, 5000, args.word_embedding_dim, num_labels, label_prop, 32, args)
        args.if_use_seq_len = 1

    print '================= model solver ...'
    # solver: __init__(self, model, train_data, test_data, **kwargs):
    solver = ModelSolver3(model, train_loader, test_loader,
                          n_epochs=args.n_epochs,
                          batch_size=args.batch_size,
                          update_rule=args.update_rule,
                          learning_rate=args.learning_rate,
                          pretrained_model=args.pretrained_model_path,
                          model_path=args.folder_path + args.model + '/',
                          log_path=args.folder_path + args.model + '/',
                          test_path=args.folder_path + args.model + '/'
                          )
    # train
    if args.train:
        print '================= begin training...'
        solver.train(args.folder_path + args.model + '/outcome.txt')

    # test
    if args.test:
        print '================= begin testing...'
        solver.test(args.folder_path + args.model + '/' + args.pretrained_model_path, args.folder_path + args.model + '/test_outcome.txt')

    # predict
    if args.predict:
        print '================= begin predicting...'
        predict_path = args.folder_path+'model_save/'+args.model+'/'
        solver.predict(trained_model_path=predict_path,
                       output_file_path=predict_path+'predict_outcome.txt',
                       k=10, emb_saved=1, can_saved=1)



if __name__ == "__main__":
    main()
