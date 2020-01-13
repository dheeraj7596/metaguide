import numpy as np


def tokenize_corpus(corpus, vocabulary):
    tokenized_corpus = []
    for x in corpus:
        tokens = x.lower().strip().split()
        trimmed_tokens = []
        for tok in tokens:
            try:
                temp = vocabulary[tok]
                trimmed_tokens.append(tok)
            except:
                continue
        tokenized_corpus.append(trimmed_tokens)
    return tokenized_corpus


def create_vocabulary(corpus, min_count=5):
    vocabulary = {}
    for row in corpus:
        tokens = row.lower().strip().split()
        for tok in tokens:
            try:
                vocabulary[tok] += 1
            except:
                vocabulary[tok] = 1

    delete_keys = []
    for i in vocabulary:
        if vocabulary[i] < min_count:
            delete_keys.append(i)

    for key in delete_keys:
        del vocabulary[key]

    word2idx = {}
    idx2word = {}
    count = 0
    for i in vocabulary:
        word2idx[i] = count
        idx2word[count] = i
        count += 1

    return vocabulary, word2idx, idx2word


def update_vocab(label_auth_dict, vocabulary, word2idx, idx2word):
    vocab_size = len(vocabulary)

    auth_set = set()
    for l in label_auth_dict:
        auth_set.update(set(label_auth_dict[l]))

    count = vocab_size
    for aut in auth_set:
        word2idx[aut] = count
        idx2word[count] = aut
        vocabulary.append(aut)
        count += 1
    return vocabulary, word2idx, idx2word
