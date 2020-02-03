from cocube_utils_beta import get_distinct_labels, train_classifier
from coc_data_utils import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from pagerank import run_author, run_conf
import pickle
import numpy as np
import sys
import math
import copy
from scipy import sparse


def get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index,
                       term_count, word_to_index, doc_freq_thresh):
    E_LT = np.zeros((label_count, term_count))
    components = {}
    for l in label_docs_dict:
        components[l] = {}
        docs = label_docs_dict[l]
        docfreq_local = calculate_doc_freq(docs)
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        rel_freq = np.sum(X_arr, axis=0) / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            try:
                if docfreq_local[name] < doc_freq_thresh:
                    continue
            except:
                continue
            E_LT[label_to_index[l]][word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[name] \
                                                           * np.tanh(rel_freq[i])
            components[l][name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                                   "idf": inv_docfreq[name],
                                   "rel_freq": np.tanh(rel_freq[i]),
                                   "rank": E_LT[label_to_index[l]][word_to_index[name]]}
    return E_LT, components


def update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict):
    word_map = {}
    for l in range(label_count):
        if not np.any(E_LT):
            n = 0
        else:
            n = min(n1 * (it + 1), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
        inds_popular = E_LT[l].argsort()[::-1][:n]

        if not np.any(F_LT):
            n = 0
        else:
            n = min(n2 * (it + 1), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
        inds_exclusive = F_LT[l].argsort()[::-1][:n]

        for word_ind in inds_popular:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if E_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], E_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], E_LT[l][word_ind])

        for word_ind in inds_exclusive:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if F_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], F_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], F_LT[l][word_ind])

    label_term_dict = defaultdict(set)
    for word in word_map:
        label, val = word_map[word]
        label_term_dict[label].add(word)
    return label_term_dict


def update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index,
                           index_to_word, inv_docfreq, docfreq, it, n1, n2, doc_freq_thresh=5):
    label_count = len(label_to_index)
    term_count = len(word_to_index)
    label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

    E_LT, components = get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict,
                                          label_to_index, term_count, word_to_index, doc_freq_thresh)
    F_LT = np.zeros((label_count, term_count))

    label_term_dict = update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict)
    return label_term_dict, components


def update_label_author_dict(label_author_dict, df, pred_labels, it, n1=7):
    label_docs_dict = get_label_docs_dict(df, label_author_dict, pred_labels)
    for l in label_author_dict:
        n = min(n1 * (it + 1), int(math.log(len(label_docs_dict[l]), 1.5)))
        label_author_dict[l] = label_author_dict[l][:n]
    return label_author_dict


def update_label_conf_dict(label_conf_dict, it):
    n = min(it + 1, 3)
    for l in label_conf_dict:
        label_conf_dict[l] = label_conf_dict[l][:n]
    return label_conf_dict


if __name__ == "__main__":
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "dblp/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_mapped_labels_phrase_removed_stopwords.pkl", "rb"))
    phrase_id_map = pickle.load(open(pkl_dump_dir + "phrase_id_map.pkl", "rb"))
    id_phrase_map = {}
    for ph in phrase_id_map:
        id_phrase_map[phrase_id_map[ph]] = ph
    tokenizer = pickle.load(open(pkl_dump_dir + "tokenizer.pkl", "rb"))
    word_to_index, index_to_word = create_index(tokenizer)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords.json")
    label_term_dict = modify_phrases(label_term_dict, phrase_id_map)
    docfreq = get_doc_freq(df)
    inv_docfreq = get_inv_doc_freq(df, docfreq)
    G_conf = sparse.load_npz(pkl_dump_dir + "G_conf.npz")
    venue_id = pickle.load(open(pkl_dump_dir + "venue_id.pkl", "rb"))
    id_venue = pickle.load(open(pkl_dump_dir + "id_venue.pkl", "rb"))
    G_auth = sparse.load_npz(pkl_dump_dir + "G_auth.npz")
    author_id = pickle.load(open(pkl_dump_dir + "author_id.pkl", "rb"))
    id_author = pickle.load(open(pkl_dump_dir + "id_author.pkl", "rb"))

    label_author_dict = {}
    label_conf_dict = {}

    t = 10
    pre_train = 0

    for i in range(t):
        print("ITERATION ", i)
        print("Going to train classifier..")

        if i == 0 and pre_train:
            pred_labels = pickle.load(open(pkl_dump_dir + "first_iteration_pred_labels.pkl", "rb"))
            probs = pickle.load(open(pkl_dump_dir + "first_iteration_probs.pkl", "rb"))

        else:
            pred_labels, probs = train_classifier(df, labels, label_term_dict, label_author_dict, label_conf_dict,
                                                  label_to_index, index_to_label, author_id, venue_id)
        label_author_dict = run_author(probs, df, G_auth, author_id, id_author, label_to_index)
        label_conf_dict = run_conf(probs, df, G_conf, venue_id, id_venue, label_to_index)

        label_author_dict = update_label_author_dict(label_author_dict, df, pred_labels, i)
        label_conf_dict = update_label_conf_dict(label_conf_dict, i)

        print("Updating label term dict..")
        label_term_dict, components = update_label_term_dict(df, label_term_dict, pred_labels, label_to_index,
                                                             index_to_label, word_to_index, index_to_word, inv_docfreq,
                                                             docfreq, i, n1=7, n2=7)
        print_label_term_dict(label_term_dict, components, id_phrase_map)
        print_label_author_dict(label_author_dict, id_author)
        print_label_conf_dict(label_conf_dict, id_venue)
        print("#" * 80)
