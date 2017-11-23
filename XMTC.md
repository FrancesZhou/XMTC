[TOC]
<!-- MarkdownTOC depth=3 -->

- Extreme Multi-label Text Classification
	- Preliminary overview
	- Preliminary attention-based model for XMTC
		- DeepWalk
		- KATE
		- Implementation
		- Problem
		- procedure
	- Analysis of raw data

<!-- /MarkdownTOC -->
# Extreme Multi-label Text Classification

## Preliminary overview

For two datasets: AmazonCat-13K and Wiki10-30K

1. AmazonCat-13K
The descriptions are not complete. But the title information is alright. Try to build the attention-based neural network model on title sequences for multi-label classification.

2. Wiki10-30K
The raw wikipedia text is alright except that you need to match the raw texts to the processed ones. It requires analyzing the raw html file. This needs lots of efforts.

Please try first on title sequences.

## Preliminary attention-based model for XMTC

Two techniques: DeepWalk and Glove

Data file: 

amazonCat_train, amazonCat_test; (contains labels for each example)

AmazonCat-13K_train_map.txt, AmazonCat-13K_test_map.txt; (contains product titles for each example)

### DeepWalk

DeepWalk python code: [github](https://github.com/phanein/deepwalk.git)

Tips:

1. sudo pip install --upgrade gensim

2. In gensim/models/word2vec.py, line 1233 or 1243, change
self.seeded_vector(self.wv.index2word[i] + str(self.seed)) to self.seeded_vector(str(self.wv.index2word[i]) + str(self.seed))

3. In deepwalk/__main__.py, line 98, change
model.save_word2vec_format(args.output) to 
model.wv.save_word2vec_format(args.output)

4. In case 'WARNING: consider setting a smaller 'batch_words' for smoother alpha decay', add 'batch_words=2000' in deepwalk/__main__.py, line 75:
model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, workers=args.workers, batch_words=2000)

### KATE

KATE code: [github](https://github.com/hugochan/KATE)

Run the code on dataset: 20-newsgroups

1. UTF-8 encoding and tokenize

See function tiny_tokenize() in preprocessing.py. [ignore error symbol when encoding UTF-8, remove word punctuation, stemming, stopwords]

2. corpus [train.corpus, test.corpus]

**vocab: **
most frequent top n words. # vocab: {'word_token': word_id, ...}
**word_freq: **
word frequency in vocab. # word_freq: {word_id: freq, ...}
**docs: **
docs with word_id and freqency. # docs: {'doc_name': {word_id: freq, ...}, ...} where the doc_name contains the classification info.

3. labels [train.labels, test.labels]

**doc_labels: **
lables for docs. # doc_labels: {'doc_name': 'label_name', ...}

### Implementation

1. corpus

**.corpus: ** docs with sequential word symbols. # docs: [[s1, s2,...], [s1, s2,...], ...]

2. labels

**.labels: ** related labels for each doc. # labels: [[l1, l2,...], [l1, l2,...], ...]

3. label embedding

For each dataset, we can get the co-currence info from train.txt

labels.pair is the pickle file that contains label pairs. \
labels.edgelist is the file used for deepwalk. (notice the format is edgelist) \
labels.embeddings is the embedding file for each label.

For AmazonCat-13K, the label dimensionality is 13330, but the number of labels which appear in labels.pair is 13326. Thus, there are 4 labels that occur separately: [5960, 390, 342, 5071] where indices are from 0 on.

### Problem

1. invalid text

There are some examples whose texts are invalid. (the words in text cannot be found in vocab) Thus, these examples should be excluded.
For AmazonCat-13K, valid/invalid train examples: 1182134/4105;
valid/invalid test examples: 305701/1081

### procedure

----------- using all titles as text -------------
when label embeddings are given, (not in trainable variables)
training for all negative labels, run 50 batches for 4 minutes. 1182134/16 /50 *4/60 = 98 h
training for limited random negative labels, run 100 batches for 70 seconds. 1182134/16 /100 *70/3600 = 14 h

first try - label embedding fixed, train 1 epoch, calculate count/all
p@1 : 0.10524
p@3 ：0.22905
p#5 ：0.32749

second try - label embedding fixed, train 1 epoch, calculate real precision and ndcg
p@1 : 0.1046	ndcg@1 : 0.1034
p@3 : 0.1032	ndcg@3 : 0.1012
p@5 : 0.1026	ndcg@5 : 0.1037

Comments-1 from Miss Shen
1. ratio between positive samples and negative samples should be fixed.
2. use descriptions instead of titles
3. decrease the dimension of label embedding, delete those separate labels

----------- using descriptions as text -------------
categories: 2674 products
descriptions: 1680 products
set(descriptions) - set(categories): 91 products. Thus, we use 1589 categories and descriptions.
FILE:
doc_data (len-1589): {'pid': 'title'+'Categories: '+categories+'Descriptions: '+descriptions, ...}
label_data (len-1589): {'pid': [label1,label2,...], 'pid': [label1,label2, ...], ...}
train_pid (len-1290): [pid0, pid1, ..., pid1289]
test_pid (len-299): [pid0, pid1, ..., pid298]

all_labels (len-1788)
all_labels_in_pair (len-1785)
separate labels: 342, 744, 5960
After removing separate labels, number of train/test examples is 1234/282, doc_data/label_data (len-1516)
FILE:
all_labels (len-1785): [label1, label2, ..., label1784]
doc_data (len-1516): {'pid': 'title'+'Categories: '+categories+'Descriptions: '+descriptions, ...}
label_data (len-1516): {'pid': [label1,label2,...], 'pid': [label1,label2, ...], ...}
train_pid (len-1234): [pid0, pid1, ..., pid1233]
test_pid (len-282): [pid0, pid1, ..., pid281]

try:
1. use descriptions, deepwalk
2. use descriptions, LINE
3. use descriptions, label embeddings represented as aggregates of word embeddings

when generating postive samples,
1) randomly choose one pid from train_pid
2) randomly choose one label from label_data[pid]
3) follow-up processing: remove the label in label_data[pid], if label_data[pid] is empty: train_pid.remove(pid), del label_data[pid].
when generating negative samples,
1) randomly choose one pid from train_pid
2) randomly choose one label from set(all_labels)-set(label_data[pid])

Optimization for tomorrow:
1. word embeddings stored as npy file (20M, 300)
2. 

## Analysis of raw data