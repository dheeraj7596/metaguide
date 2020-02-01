from cocube_utils import get_distinct_labels, train_classifier
from coc_data_utils import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pickle
import numpy as np
import sys
import math
import copy


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


def get_exclusive_matrix(doc_freq_thresh, index_to_label, index_to_word, inv_docfreq, label_count, label_docs_dict,
                         label_to_index, term_count, word_to_index):
    E_LT = np.zeros((label_count, term_count))
    components = {}
    for l in label_docs_dict:
        components[l] = {}
    for l in label_docs_dict:
        docs = label_docs_dict[l]
        docfreq = calculate_doc_freq(docs)
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        freq = np.sum(X_arr, axis=0)
        rel_freq = freq / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            try:
                if docfreq[name] < doc_freq_thresh:
                    continue
            except:
                continue
            E_LT[label_to_index[l]][word_to_index[name]] = (rel_freq[i] ** 0.2) * (freq[i] ** 1.5)
            components[l][name] = {"relfreq": rel_freq[i] ** 0.2, "freq": freq[i] ** 1.5}
    for l in range(label_count):
        zero_counter = 0
        for t in range(term_count):
            flag = 0
            if E_LT[l][t] == 0:
                continue
            col_list = list(E_LT[:, t])
            temp_list = copy.deepcopy(col_list)
            temp_list.pop(l)
            den = np.nanmax(temp_list)
            if den == 0:
                flag = 1
                den = 0.0001
                zero_counter += 1
            temp = E_LT[l][t] / (den ** 0.2)
            E_LT[l][t] = temp * inv_docfreq[index_to_word[t]]
            components[index_to_label[l]][index_to_word[t]]["ratio"] = components[index_to_label[l]][index_to_word[t]][
                                                                           "relfreq"] / (den ** 0.2)
            components[index_to_label[l]][index_to_word[t]]["idf"] = inv_docfreq[index_to_word[t]]
            components[index_to_label[l]][index_to_word[t]]["rare"] = flag
            components[index_to_label[l]][index_to_word[t]]["rank"] = E_LT[l][t]
        print(index_to_label[l], zero_counter)
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


if __name__ == "__main__":
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "dblp/"
    pkl_dump_dir = basepath + dataset

    pre_trained = 0

    df = pickle.load(open(pkl_dump_dir + "df_mapped_labels_phrase.pkl", "rb"))
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

    # df = modify_df(df, docfreq, 5)
    t = 10

    for i in range(t):
        print("ITERATION ", i)
        print("Going to train classifier..")
        if i == 0 and pre_trained == 1:
            pred_labels = pickle.load(open(pkl_dump_dir + "seedwords_pred.pkl", "rb"))
        else:
            pred_labels = train_classifier(df, labels, label_term_dict, label_to_index, index_to_label)
        # if i == 0:
        #     pickle.dump(pred_labels, open(pkl_dump_dir + "seedwords_pred.pkl", "wb"))
        print("Updating label term dict..")
        label_term_dict, components = update_label_term_dict(df, label_term_dict, pred_labels, label_to_index,
                                                             index_to_label, word_to_index, index_to_word, inv_docfreq,
                                                             docfreq, i, n1=7, n2=7)
        print_label_term_dict(label_term_dict, components, id_phrase_map)
        print("#" * 80)
