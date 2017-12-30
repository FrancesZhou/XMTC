'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import re
import numpy as np
import cPickle as pickle

def dump_pickle(data, file):
    try:
        with open(file, 'w') as datafile:
            pickle.dump(data, datafile)
    except Exception as e:
        raise e

def load_pickle(file):
    try:
        with open(file, 'r') as datafile:
            data = pickle.load(datafile)
    except Exception as e:
        raise e
    return data

def write_file(data, file):
    try:
        with open(file, 'w') as datafile:
            for line in data:
                datafile.write(str(line[0]) + '\t' + str(line[1]) + '\n')
    except Exception as e:
        raise e

def load_txt(file):
    try:
        with open(file, 'r') as df:
            data = df.readlines()
    except Exception as e:
        raise e
    return data

def not_empty(s):
    #return s.strip()
    return s and s.strip()

# input: a text with several words, vocab
# output: a list consisted of wordIDs in vocab
def get_wordID_from_vocab(text, vocab):
    all_tokens = re.split('([A-Za-z]+)|([0-9]+)|(\W)', text)
    all_tokens = filter(not_empty, all_tokens)
    all_tokens = [e.strip() for e in all_tokens]
    # check if tokens are in the vocab
    token_indices = []
    for t in all_tokens:
        try:
            ind = vocab.index(t)
            token_indices.append(ind)
        except:
            continue
    return token_indices


def write_label_pairs_into_file(label_pairs, output_file):
    txtfile = open(output_file, 'w')
    #label_pairs = load_pickle(label_pairs_file)
    for i in range(len(label_pairs)):
        txtfile.write(str(label_pairs[i][0]) + '\t' + str(label_pairs[i][1]))
        txtfile.write('\n')

