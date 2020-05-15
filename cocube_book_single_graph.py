from cocube_utils_beta_books import get_distinct_labels, train_classifier, get_entity_count, plot_entity_count, \
    get_cut_off
from cocube_variations import *
from pagerank import run_pagerank_single_graph
import pickle
import os
from scipy import sparse
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)


def modify(label_term_dict):
    for l in label_term_dict:
        temp = {}
        for t in label_term_dict[l]:
            temp[t] = 1
        label_term_dict[l] = temp
    return label_term_dict


if __name__ == "__main__":
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "books/"
    pkl_dump_dir = basepath + dataset
    model_name = sys.argv[1]
    is_soft = 0

    df = pickle.load(open(pkl_dump_dir + "df_phrase_removed_stopwords.pkl", "rb"))
    phrase_id_map = pickle.load(open(pkl_dump_dir + "phrase_id_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(pkl_dump_dir + "id_phrase_map.pkl", "rb"))
    tokenizer = pickle.load(open(pkl_dump_dir + "tokenizer.pkl", "rb"))
    word_to_index, index_to_word = create_index(tokenizer)
    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords.json")
    label_term_dict = modify_phrases(label_term_dict, phrase_id_map)
    docfreq = get_doc_freq(df)
    inv_docfreq = get_inv_doc_freq(df, docfreq)

    G_all = sparse.load_npz(pkl_dump_dir + "G_all.npz")

    fnust_id = pickle.load(open(pkl_dump_dir + "fnust_id_all.pkl", "rb"))
    id_fnust = pickle.load(open(pkl_dump_dir + "id_fnust_all.pkl", "rb"))
    author_id = pickle.load(open(pkl_dump_dir + "author_id_all.pkl", "rb"))
    id_author = pickle.load(open(pkl_dump_dir + "id_author_all.pkl", "rb"))
    pub_id = pickle.load(open(pkl_dump_dir + "pub_id_all.pkl", "rb"))
    id_pub = pickle.load(open(pkl_dump_dir + "id_pub_all.pkl", "rb"))
    author_pub_id = pickle.load(open(pkl_dump_dir + "author_pub_id_all.pkl", "rb"))
    id_author_pub = pickle.load(open(pkl_dump_dir + "id_author_pub_all.pkl", "rb"))
    pub_year_id = pickle.load(open(pkl_dump_dir + "pub_year_id_all.pkl", "rb"))
    id_pub_year = pickle.load(open(pkl_dump_dir + "id_pub_year_all.pkl", "rb"))

    phrase_docid_map = pickle.load(open(pkl_dump_dir + "phrase_docid_map.pkl", "rb"))
    author_docid_map = pickle.load(open(pkl_dump_dir + "author_docid_map.pkl", "rb"))
    pub_docid_map = pickle.load(open(pkl_dump_dir + "pub_docid_map.pkl", "rb"))
    author_pub_docid_map = pickle.load(open(pkl_dump_dir + "author_pub_docid_map.pkl", "rb"))
    pub_year_docid_map = pickle.load(open(pkl_dump_dir + "pub_year_docid_map.pkl", "rb"))

    label_phrase_dict = modify(label_term_dict)
    print_label_phrase_dict(label_phrase_dict, id_phrase_map)
    label_author_dict = {}
    label_pub_dict = {}
    label_author_pub_dict = {}
    label_pub_year_dict = {}

    t = 9
    pre_train = 0
    should_print = True

    phrase_count = {}
    author_count = {}
    phrase_ppr_cutoff = {}
    author_ppr_cutoff = {}

    for i in range(t):
        print("ITERATION ", i)
        print("Going to train classifier..")

        if i == 0 and pre_train:
            pred_labels = pickle.load(open(pkl_dump_dir + "first_iteration_pred_labels.pkl", "rb"))
            probs = pickle.load(open(pkl_dump_dir + "first_iteration_probs.pkl", "rb"))
        # elif i == 0:
        #     pred_labels, probs = train_classifier(df, labels, label_phrase_dict, label_author_dict, label_conf_dict,
        #                                           label_to_index, index_to_label, model_name, old=True)
        else:
            pred_labels, probs = train_classifier(df, labels, label_phrase_dict, label_author_dict, label_pub_dict,
                                                  label_author_pub_dict, label_pub_year_dict, label_to_index,
                                                  index_to_label, model_name, old=True, soft=is_soft)

        phrase_plot_dump_dir = pkl_dump_dir + "images/" + model_name + "/phrase/" + str(i) + "/"
        auth_plot_dump_dir = pkl_dump_dir + "images/" + model_name + "/author/" + str(i) + "/"

        entity_id_list = [fnust_id, author_id, pub_id, author_pub_id, pub_year_id]
        id_entity_list = [id_fnust, id_author, id_pub, id_author_pub, id_pub_year]
        label_phrase_dict, label_author_dict, label_pub_dict, label_author_pub_dict, label_pub_year_dict = run_pagerank_single_graph(
            probs, df, G_all, entity_id_list, id_entity_list, label_to_index, None, plot=False)

        label_entity_dict_list = [label_phrase_dict, label_author_dict, label_pub_dict, label_author_pub_dict,
                                  label_pub_year_dict]
        entity_docid_map_list = [phrase_docid_map, author_docid_map, pub_docid_map, author_pub_docid_map,
                                 pub_year_docid_map]
        label_phrase_dict, label_author_dict, label_pub_dict, label_author_pub_dict, label_pub_year_dict = rank_phrase_metadata_together(
            label_entity_dict_list, entity_docid_map_list, df, labels, i, cov="full")

        if should_print:
            print_label_phrase_dict(label_phrase_dict, id_phrase_map)
            print_label_entity_dict(label_author_dict)
            print_label_entity_dict(label_pub_dict)
            print_label_entity_dict(label_author_pub_dict)
            print_label_entity_dict(label_pub_year_dict)
        print("#" * 80)
