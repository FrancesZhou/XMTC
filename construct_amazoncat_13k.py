'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import json
import numpy as np
from biLSTM.preprocessing.preprocessing import get_max_num_labels, generate_label_vector_of_fixed_length, construct_train_test_corpus, generate_labels_from_file_and_error, generate_label_pair_from_file
from biLSTM.utils.io_utils import load_pickle, dump_pickle, write_file

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
    #cat_file = 'datasets/AmazonCat-13K/RawData/categories.txt'
    doc_data_file = 'AmazonCat-13K/output/descriptions/doc_data.json'
    label_data_file = 'AmazonCat-13K/output/descriptions/label_data.json'
    with open(doc_data_file, 'w') as file:
        json.dump(doc_data, file)
    with open(label_data_file, 'w') as file:
        json.dump(label_data, file)

def generate_label_pair_from_file():
    with open('AmazonCat-13K/output/descriptions/label_data.json', 'r') as file:
        label_data = json.load(file)
    all_labels = []
    # get label pairs
    label_pairs = []
    for _, labels_doc in label_data.items:
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

    all_labels = np.unique(np.concatenate(all_labels))
    all_label_pair = np.unique(np.concatenate(label_pairs))
    separate_labels = list(set(all_labels) - set(all_label_pair))
    print len(separate_labels)
    return all_labels, label_pairs

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

    # train_corpus: 'datasets/AmazonCat-13K/rawdata/AmazonCat-13K_train_map.txt'
    # test_corpus: 'datasets/AmazonCat-13K/rawdata/AmazonCat-13K_test_map.txt'
    # train_labels: 'datasets/AmazonCat-13K/rawdata/amazonCat_train.txt'
    # test_labels: 'datasets/AmazonCat-13K/rawdata/amazonCat_test.txt'
    # out_dir: 'datasets/AmazonCat-13K/output/'
    ## ----------- get train/test corpus -----------
    # vocab = load_pickle(args.vocab_path)
    preprocessing_for_descriptions()
    generate_label_pair_from_file()

    # train_labels = load_pickle(os.path.join(args.out_dir, 'train.labels'))
    # test_labels = load_pickle(os.path.join(args.out_dir, 'test.labels'))
    # max_num_labels, mean_num_labels = get_max_num_labels(train_labels)
    # max_num_labels2, mean_num_labels2 = get_max_num_labels(test_labels)
    # num_labels = int(max_num_labels + mean_num_labels) + 1
    # for i in range(10):
    #     pos_labels = train_labels[i]
    #     indices, labels = generate_label_vector_of_fixed_length(pos_labels, num_labels, 13330)
    #     print indices
    #     print labels



if __name__ == "__main__":
    main()