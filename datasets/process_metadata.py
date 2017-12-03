'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import ast
import json
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

all_asin = []
categories = {}
title = {}
description = {}
others = {}

all_asin_file = 'metadata/all_asin.pkl'
categories_file = 'metadata/categories.pkl'
title_file = 'metadata/title.pkl'
description_file = 'metadata/description.pkl'
others_file = 'metadata/others.pkl'

meta_file = open('metadata.json', 'r')
all_data = meta_file.readlines()
i = 0
for line in all_data:
    if i % 50000 == 0:
        print i
    i += 1
    d = ast.literal_eval(line)
    asin = None
    for k, v in d.items():
        if k == 'asin':
            asin = v
            all_asin.append(asin)
        elif k == 'categories':
            if asin is not None:
                categories[asin] = v
            else:
                print 'error'
                print k
                print v
        elif k == 'title':
            if asin is not None:
                title[asin] = v
            else:
                print 'error'
                print k
                print v
        elif k == 'description':
            if asin is not None:
                description[asin] = v
            else:
                print 'error'
                print k
                print v
        else:
            if asin is not None:
                others[asin] = v
            else:
                print 'error'
                print k
                print v

dump_pickle(all_asin, all_asin_file)
dump_pickle(categories, categories_file)
dump_pickle(title, title_file)
dump_pickle(description, description_file)
dump_pickle(others, others_file)

