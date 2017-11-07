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

## Analysis of raw data