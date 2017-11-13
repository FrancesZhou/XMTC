'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import
import cPickle as pickle
import re
import numpy as np

def not_empty(s):
    return s and s.strip()

def gen_word_emb_from_str(str):
    _, v_str = str.split(' ', 1)
    v = [float(e) for e in v_str.split()]
    return v

# indices of missing labels : [5960, 390, 342, 5071] where indices are from 0 on.
# missing = ['all electronics', 'amazon instant video', 'furniture & decor', 'home improvement']

# with open('../../../dataset/vocab', 'r') as df:
#     vocab = pickle.load(df)
#
# print vocab.index('decor')

# missing_label_tokens = []
#
# for txt in missing:
#     all_tokens = re.split('([A-Za-z]+)|([0-9]+)', txt)
#     print all_tokens
#     all_tokens = filter(not_empty, all_tokens)
#     missing_label_tokens.append(all_tokens)
#missing_index = [5960, 390, 342, 5071]
missing_index = [342, 391, 5071, 5960]
missing_label_tokens = [['all', 'electronics'], ['amazon', 'instant', 'video'], ['furniture', '&', 'decor'], ['home', 'improvement']]

with open('labels.embeddings', 'r') as df:
    label_embeddings = df.readlines()
with open('vocab', 'r') as df:
    vocab = pickle.load(df)
print 'begin to read word_embeddings in glove'
with open('glove.840B.300d.txt', 'r') as df:
    word_embeddings = df.readlines()

print 'loaded word_embeddings'
label_embs = []
for missing_label in missing_label_tokens:
    emb = []
    for word in missing_label:
        ind = vocab.index(word)
        emb.append(gen_word_emb_from_str(word_embeddings[ind]))
    emb = np.mean(emb, axis=0)
    label_embs.append(emb)

txtfile = open('missinglabels.embeddings', 'w')
for i in range(len(missing_index)):
    txtfile.write(str(missing_index[i])+' ')
    for f in label_embs[i]:
        txtfile.write(str(f)+' ')
    txtfile.write('\n')
    #all_label_embeddings = np.insert(label_embeddings, missing_index[i], )



