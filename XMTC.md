[TOC]
<!-- MarkdownTOC depth=3 -->

- Extreme Multi-label Text Classification
	- Preliminary overview
	- Preliminary attention-based model for XMTC
		- DeepWalk
		- KATE
		- Implementation
		- Problem
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

## Analysis of raw data