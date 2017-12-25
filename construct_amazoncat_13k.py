'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import json
import numpy as np
from model.preprocessing.preprocessing import get_wordID_from_vocab, gen_word_emb_from_str, construct_train_test_corpus, generate_labels_from_file_and_error, generate_label_pair_from_file
from model.utils.io_utils import load_pickle, dump_pickle, load_txt

def preprocessing_for_all_titles(args, vocab):
    train_corpus, test_corpus = construct_train_test_corpus(vocab, args.train_corpus_path, args.test_corpus_path, args.out_dir)
    # ----------- get train/test labels -----------
    train_labels = generate_labels_from_file_and_error(args.train_labels_path,
                                                       os.path.join(args.out_dir, 'train_error.index'),
                                                       os.path.join(args.out_dir, 'train.labels'))
    test_labels = generate_labels_from_file_and_error(args.test_labels_path, os.path.join(args.out_dir, 'test_error.index'),
                                                      os.path.join(args.out_dir, 'test.labels'))
    #----------- generate label pairs(used by deepwalk) ----------
    if args.if_label_pair:
        label_pairs = generate_label_pair_from_file(os.path.join(args.out_dir, 'train.labels'),
                                                    os.path.join(args.out_dir, 'labels.pair'))
        #------------ analysis of labels - -----------
        label_pairs = load_pickle(os.path.join(args.out_dir, 'labels.pair'))
        l = label_pairs.flatten()
        l = np.unique(l)
        print len(l)
        print max(l)
        print min(l)
        # find those separate labels
        all_labels = load_pickle(os.path.join(args.out_dir, 'train.labels'))
        all_labels = np.hstack(all_labels)
        all_labels = np.unique(all_labels)
        print len(all_labels)
        print max(all_labels)
        print min(all_labels)
        print(set(all_labels) - set(l))
        print 'done.'

def preprocessing_for_descriptions():
    des_file = 'datasets/AmazonCat-13K/RawData/descriptions.txt'
    cat_file = 'datasets/AmazonCat-13K/RawData/categories.txt'
    title_train_file = 'datasets/AmazonCat-13K/ProcessedData/AmazonCat-13K_train_map.txt'
    title_test_file = 'datasets/AmazonCat-13K/ProcessedData/AmazonCat-13K_test_map.txt'
    label_train_file = 'datasets/AmazonCat-13K/ProcessedData/amazonCat_train.txt'
    label_test_file = 'datasets/AmazonCat-13K/ProcessedData/amazonCat_test.txt'
    # corpus: titles: cat1; cat2; cat3. Descriptions
    # des_data
    with open(des_file) as file:
        des_lines = file.readlines()
        i = 0
        des_data = {}
        #print des_lines[-3:]
        while i<len(des_lines):
            pid = des_lines[i].split()[-1]
            des = des_lines[i+1].split(' ', 1)[-1]
            des_data[pid] = des
            try:
                if des_lines[i+2] == '\n':
                    i += 3
                else:
                    print 'error!'
            except Exception as e:
                if i+2 == len(des_lines):
                    break
                else:
                    raise e
    # cat_data
    with open(cat_file) as file:
        cat_lines = file.readlines()
        i = 0
        cat_data = {}
        #print cat_lines[-5:]
        while i<len(cat_lines):
            pid = cat_lines[i].strip()
            i += 1
            cats = []
            while True:
                try:
                    line = cat_lines[i]
                except Exception as e:
                    if i == len(cat_lines):
                        break
                    else:
                        raise e
                if line[0] == ' ':
                    cats.append(line.strip())
                    i += 1
                else:
                    cat_data[pid] = cats
                    break
    no_ids = list(set(des_data.keys()) - set(cat_data.keys()))
    print len(no_ids)
    for no_id in no_ids:
        del des_data[no_id]
    #print len(list(set(des_data.keys()) - set(cat_data.keys())))
    no_ids_cat = list(set(cat_data.keys()) - set(des_data.keys()))
    for no_id in no_ids_cat:
        del cat_data[no_id]

    # train_id_index
    doc_data = {}
    pids = des_data.keys()
    with open(title_train_file) as file:
        title_train_lines = file.readlines()
        train_id_index = {}
        for i in range(len(title_train_lines)):
            if len(pids) == 0:
                break
            pid, title = title_train_lines[i].split('->', 1)
            if pid in pids:
                text = title.strip() + '. Categories: '
                cats = cat_data[pid]
                for j in range(len(cats)):
                    if j == len(cats) - 1:
                        text = text + cats[j] + '. '
                    else:
                        text = text + cats[j] + '; '
                des = des_data[pid]
                text = text + 'Description: ' + des
                doc_data[pid] = text
                train_id_index[pid] = i
                pids.remove(pid)
    # test_id_index
    with open(title_test_file) as file:
        title_test_lines = file.readlines()
        test_id_index = {}
        for i in range(len(title_test_lines)):
            if len(pids) == 0:
                break
            pid, title = title_test_lines[i].split('->', 1)
            if pid in pids:
                text = title.strip() + '. Categories: '
                cats = cat_data[pid]
                for j in range(len(cats)):
                    if j == len(cats) - 1:
                        text = text + cats[j] + '. '
                    else:
                        text = text + cats[j] + '; '
                des = des_data[pid]
                text = text + 'Descriptions: ' + des
                doc_data[pid] = text
                test_id_index[pid] = i
                pids.remove(pid)
    # no_id_title = list(set(des_data.keys()) - set(title_train_data.keys()) - set(title_test_data.keys()))
    # print len(no_id_title)
    # if len(no_id_title):
    #     for no_id in no_id_title:
    #         del des_data[no_id]
    # print len(list(set(des_data.keys()) - set(title_train_data.keys()) - set(title_test_data.keys())))

    label_data = {}
    with open(label_train_file) as file:
        label_train_lines = file.readlines()
        for k, v in train_id_index.items():
            labels_str = label_train_lines[v+1].split(' ', 1)[0]
            labels_str = labels_str.split(',')
            labels_doc = [int(label) for label in labels_str]
            label_data[k] = labels_doc

    with open(label_test_file) as file:
        label_test_lines = file.readlines()
        for k, v in test_id_index.items():
            labels_str = label_test_lines[v+1].split(' ', 1)[0]
            labels_str = labels_str.split(',')
            labels_doc = [int(label) for label in labels_str]
            label_data[k] = labels_doc

    # save doc_data, label_data
    train_pid = train_id_index.keys()
    test_pid = test_id_index.keys()
    doc_data_file = 'datasets/AmazonCat-13K/output/descriptions/doc_data.pkl'
    label_data_file = 'datasets/AmazonCat-13K/output/descriptions/label_data.pkl'
    train_pid_file = 'datasets/AmazonCat-13K/output/descriptions/train_pid.pkl'
    test_pid_file = 'datasets/AmazonCat-13K/output/descriptions/test_pid.pkl'
    dump_pickle(doc_data, doc_data_file)
    dump_pickle(label_data, label_data_file)
    dump_pickle(train_pid, train_pid_file)
    dump_pickle(test_pid, test_pid_file)

def generate_label_pair_from_file():
    file = 'datasets/AmazonCat-13K/output/descriptions/label_data.pkl'
    label_data = load_pickle(file)
    all_labels = []
    # get label pairs
    label_pairs = []
    for _, labels_doc in label_data.items():
        all_labels.append(labels_doc)
        if len(labels_doc) == 1:
            continue
        labels_doc = sorted(labels_doc)
        label_pair_start = labels_doc[0]
        for label in labels_doc[1:]:
            label_pairs.append([label_pair_start, label])
    # delete duplica
    label_pairs = np.array(label_pairs, dtype=np.int32)
    label_pairs = np.unique(label_pairs, axis=0)

    all_labels = np.concatenate(all_labels)
    all_labels = np.unique(all_labels)
    all_label_pair = np.unique(np.concatenate(label_pairs))
    separate_labels = list(set(all_labels) - set(all_label_pair))
    print len(separate_labels)
    dump_pickle(label_pairs, 'datasets/AmazonCat-13K/output/descriptions/label_pair/labels_pair.pkl')
    dump_pickle(all_label_pair, 'datasets/AmazonCat-13K/output/descriptions/label_pair/all_labels.pkl')
    return all_label_pair, separate_labels

def write_label_pair():
    txtfile = open('datasets/AmazonCat-13K/output/descriptions/label_pair/labels.edgelist', 'w')
    label_pairs = load_pickle('datasets/AmazonCat-13K/output/descriptions/label_pair/labels_pair.pkl')
    for i in range(len(label_pairs)):
        txtfile.write(str(label_pairs[i][0]) + '\t' + str(label_pairs[i][1]))
        txtfile.write('\n')

def get_valid_doc_label_data(separate_labels):
    doc_data_file = 'datasets/AmazonCat-13K/output/descriptions/doc_data.pkl'
    label_data_file = 'datasets/AmazonCat-13K/output/descriptions/label_data.pkl'
    train_pid_file = 'datasets/AmazonCat-13K/output/descriptions/train_pid.pkl'
    test_pid_file = 'datasets/AmazonCat-13K/output/descriptions/test_pid.pkl'
    doc_data = load_pickle(doc_data_file)
    label_data = load_pickle(label_data_file)
    train_pid = load_pickle(train_pid_file)
    test_pid = load_pickle(test_pid_file)
    for pid, l in label_data.items():
        l2 = list(set(l) - set(separate_labels))
        if len(l2) > 1:
            label_data[pid] = l2
        else:
            del label_data[pid]
            del doc_data[pid]
            if pid in train_pid:
                train_pid.remove(pid)
            elif pid in test_pid:
                test_pid.remove(pid)
            else:
                print 'error!'
    # for l in separate_labels:
    #     all_labels.remove(l)
    dump_pickle(doc_data, 'datasets/AmazonCat-13K/output/descriptions/label_pair/doc_data.pkl')
    dump_pickle(label_data, 'datasets/AmazonCat-13K/output/descriptions/label_pair/label_data.pkl')
    dump_pickle(train_pid, 'datasets/AmazonCat-13K/output/descriptions/label_pair/train_pid.pkl')
    dump_pickle(test_pid, 'datasets/AmazonCat-13K/output/descriptions/label_pair/test_pid.pkl')

def get_doc_wordID_data_from_vocab(vocab):
    doc_data = load_pickle('datasets/AmazonCat-13K/output/descriptions/label_pair/doc_data.pkl')
    doc_token_data = {}
    count = 0
    for pid, seq in doc_data.items():
        count += 1
        if count % 50 == 0:
            print count
        token_indices = get_wordID_from_vocab(seq, vocab)
        doc_token_data[pid] = token_indices
    doc_wordID_data_file = 'datasets/AmazonCat-13K/output/descriptions/label_pair/doc_wordID_data.pkl'
    dump_pickle(doc_token_data, doc_wordID_data_file)

def process_word_embeddings(file):
    txt_word_embeddings = load_txt(file)
    word_embeddings = []
    count = 0
    for line in txt_word_embeddings:
        if count % 1000 == 0:
            print count
        emb = gen_word_emb_from_str(line)
        word_embeddings.append(emb)
        count += 1
    word_embeddings = np.array(word_embeddings, dtype=np.float32)
    np.save('datasets/word_embeddings.npy', word_embeddings)

def get_train_test_pos_samples():
    label_data = load_pickle('datasets/AmazonCat-13K/output/descriptions/label_pair/label_data.pkl')
    train_pid = load_pickle('datasets/AmazonCat-13K/output/descriptions/label_pair/train_pid.pkl')
    test_pid = load_pickle('datasets/AmazonCat-13K/output/descriptions/label_pair/test_pid.pkl')
    train_pos_samples = []
    test_pos_samples = []
    for pid in train_pid:
        train_pos_samples.append(label_data[pid])
    for pid in test_pid:
        test_pos_samples.append(label_data[pid])
    train_pos_samples = np.concatenate(train_pos_samples)
    test_pos_samples = np.concatenate(test_pos_samples)
    print len(train_pos_samples)
    print len(test_pos_samples)

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-train_corpus', '--train_corpus_path', type=str,
                       default='datasets/AmazonCat-13K/rawdata/AmazonCat-13K_train_map.txt',
                       help='path to the training corpus')
    parse.add_argument('-test_corpus', '--test_corpus_path', type=str,
                       default='datasets/AmazonCat-13K/rawdata/AmazonCat-13K_test_map.txt',
                       help='path to the testing corpus')
    parse.add_argument('-train_labels', '--train_labels_path', type=str,
                       default='datasets/AmazonCat-13K/rawdata/amazonCat_train.txt',
                       help='path to the training labels')
    parse.add_argument('-test_labels', '--test_labels_path', type=str,
                       default='datasets/AmazonCat-13K/rawdata/amazonCat_test.txt',
                       help='path to the testing labels')
    parse.add_argument('-o', '--out_dir', type=str,
                       default='datasets/AmazonCat-13K/output/',
                       help='path to the output dir')
    parse.add_argument('-if_label_pair', '--if_label_pair', type=bool, default=True, help='whether to generate label pairs')

    parse.add_argument('-vocab', '--vocab_path', type=str,
                       default='datasets/vocab',
                       help='path to the testing labels')
    args = parse.parse_args()

    ## ----------- get train/test corpus -----------
    # preprocessing_for_descriptions()
    #_, separate_labels = generate_label_pair_from_file()
    # separate_labels = [342, 744, 5960]
    # write_label_pair()
    # get_valid_doc_label_data(separate_labels)
    # label_embeddings = generate_label_embedding_from_file('datasets/AmazonCat-13K/output/descriptions/label_pair/label.embeddings')
    # dump_pickle(label_embeddings, 'datasets/AmazonCat-13K/output/descriptions/label_pair/label_embeddings.pkl')
    #vocab = load_pickle(args.vocab_path)
    #get_doc_wordID_data_from_vocab(vocab)
    #get_train_test_pos_samples()
    process_word_embeddings('datasets/glove.840B.300d.txt')



if __name__ == "__main__":
    main()
