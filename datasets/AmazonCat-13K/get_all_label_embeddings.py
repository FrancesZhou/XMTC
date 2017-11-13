'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import
import cPickle as pickle
#import re
import numpy as np

with open('labels.embeddings', 'r') as df:
    txt_label_embeddings = df.readlines()

with open('missinglabels.embeddings', 'r') as df:
    txt_missing_label_embeddings = df.readlines()

num_label = len(txt_label_embeddings) + len(txt_missing_label_embeddings) - 1
print num_label
txt_header = txt_label_embeddings[0]
dw_num, emb_dim = txt_header.split(' ')
emb_dim = int(emb_dim)
dw_num = int(dw_num)
print dw_num

label_embeddings = np.zeros([num_label, emb_dim])

for i in range(1, len(txt_label_embeddings)):
    index, v_str = txt_label_embeddings[i].split(' ', 1)
    vector = [float(e) for e in v_str.split()]
    label_embeddings[int(index)] = vector

for i in range(len(txt_missing_label_embeddings)):
    index, v_str = txt_missing_label_embeddings[i].split(' ', 1)
    vector = [float(e) for e in v_str.split()]
    label_embeddings[int(index)] = vector

with open('all_labels.embeddings', 'w') as df:
    pickle.dump(label_embeddings, df)