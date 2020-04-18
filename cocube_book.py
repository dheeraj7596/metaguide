from cocube_utils_beta_books import get_distinct_labels, train_classifier, get_entity_count, plot_entity_count, \
    get_cut_off
from cocube_variations import *
from pagerank import run_pagerank
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
    model_name = sys.argv[2]
    is_soft = int(sys.argv[3])

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

    G_phrase = sparse.load_npz(pkl_dump_dir + "G_phrase.npz")
    G_auth = sparse.load_npz(pkl_dump_dir + "G_auth.npz")
    G_pub = sparse.load_npz(pkl_dump_dir + "G_pub.npz")
    G_auth_pub = sparse.load_npz(pkl_dump_dir + "G_auth_pub.npz")

    fnust_id = pickle.load(open(pkl_dump_dir + "fnust_id.pkl", "rb"))
    id_fnust = pickle.load(open(pkl_dump_dir + "id_fnust.pkl", "rb"))
    author_id = pickle.load(open(pkl_dump_dir + "author_id.pkl", "rb"))
    id_author = pickle.load(open(pkl_dump_dir + "id_author.pkl", "rb"))
    pub_id = pickle.load(open(pkl_dump_dir + "pub_id.pkl", "rb"))
    id_pub = pickle.load(open(pkl_dump_dir + "id_pub.pkl", "rb"))
    author_pub_id = pickle.load(open(pkl_dump_dir + "author_pub_id.pkl", "rb"))
    id_author_pub = pickle.load(open(pkl_dump_dir + "id_author_pub.pkl", "rb"))

    phrase_docid_map = pickle.load(open(pkl_dump_dir + "phrase_docid_map.pkl", "rb"))
    author_docid_map = pickle.load(open(pkl_dump_dir + "author_docid_map.pkl", "rb"))
    pub_docid_map = pickle.load(open(pkl_dump_dir + "pub_docid_map.pkl", "rb"))
    author_pub_docid_map = pickle.load(open(pkl_dump_dir + "author_pub_docid_map.pkl", "rb"))

    label_phrase_dict = modify(label_term_dict)
    print_label_phrase_dict(label_phrase_dict, id_phrase_map)
    label_author_dict = {}
    label_pub_dict = {}
    label_author_pub_dict = {}

    t = 9
    pre_train = 0
    plot = False
    should_print = True
    algo = int(sys.argv[1])

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
                                                  label_author_pub_dict, label_to_index,
                                                  index_to_label, model_name, old=True, soft=is_soft)

        phrase_plot_dump_dir = pkl_dump_dir + "images/" + model_name + "/phrase/" + str(i) + "/"
        auth_plot_dump_dir = pkl_dump_dir + "images/" + model_name + "/author/" + str(i) + "/"

        if plot:
            os.makedirs(phrase_plot_dump_dir, exist_ok=True)
            os.makedirs(auth_plot_dump_dir, exist_ok=True)

        # RANKING PHRASE ONLY
        if algo == 1:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             phrase_plot_dump_dir,
                                             plot=plot)
            label_phrase_dict = rank_phrase_only(label_phrase_dict, phrase_docid_map, df, labels, i)

        # RANKING AUTHOR ONLY
        elif algo == 2:
            label_author_dict = run_pagerank(probs, df, G_auth, author_id, id_author, label_to_index,
                                             auth_plot_dump_dir,
                                             plot=plot)
            label_author_dict = rank_author_only(label_author_dict, author_docid_map, df, labels, i)

        # RANKING PHRASE, METADATA TOGETHER
        elif algo == 3:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             phrase_plot_dump_dir,
                                             plot=plot)
            label_author_dict = run_pagerank(probs, df, G_auth, author_id, id_author, label_to_index,
                                             auth_plot_dump_dir,
                                             plot=plot)
            label_phrase_dict, label_author_dict = rank_phrase_author_together(label_phrase_dict, label_author_dict,
                                                                               phrase_docid_map, author_docid_map, df,
                                                                               labels, i)

        # RANKING PHRASE, METADATA TOGETHER ITERATIVE COVERAGE
        elif algo == 4:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             phrase_plot_dump_dir,
                                             plot=plot)
            label_author_dict = run_pagerank(probs, df, G_auth, author_id, id_author, label_to_index,
                                             auth_plot_dump_dir,
                                             plot=plot)
            label_phrase_dict, label_author_dict = rank_phrase_author_together(label_phrase_dict, label_author_dict,
                                                                               phrase_docid_map, author_docid_map, df,
                                                                               labels, i, cov="iterative")

        # RANKING INDEPENDENTLY
        elif algo == 5:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             phrase_plot_dump_dir,
                                             plot=plot)
            label_author_dict = run_pagerank(probs, df, G_auth, author_id, id_author, label_to_index,
                                             auth_plot_dump_dir,
                                             plot=plot)
            label_phrase_dict, label_author_dict = rank_phrase_author_independently(label_phrase_dict,
                                                                                    label_author_dict, phrase_docid_map,
                                                                                    author_docid_map, df, i)


        # RANKING PHRASE, AUTHOR, PUBLISHER TOGETHER
        elif algo == 6:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             phrase_plot_dump_dir,
                                             plot=plot)
            label_author_dict = run_pagerank(probs, df, G_auth, author_id, id_author, label_to_index,
                                             auth_plot_dump_dir,
                                             plot=plot)
            label_pub_dict = run_pagerank(probs, df, G_pub, pub_id, id_pub, label_to_index,
                                          auth_plot_dump_dir,
                                          plot=plot)
            label_phrase_dict, label_author_dict, label_pub_dict = rank_phrase_author_attr_together(label_phrase_dict,
                                                                                                    label_author_dict,
                                                                                                    label_pub_dict,
                                                                                                    phrase_docid_map,
                                                                                                    author_docid_map,
                                                                                                    pub_docid_map, df,
                                                                                                    labels, i)

        # RANKING PHRASE, AUTHOR, PUBLISHER, AUTHOR_PUBLISHER TOGETHER
        elif algo == 7:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             phrase_plot_dump_dir,
                                             plot=plot)
            label_author_dict = run_pagerank(probs, df, G_auth, author_id, id_author, label_to_index,
                                             auth_plot_dump_dir,
                                             plot=plot)
            label_pub_dict = run_pagerank(probs, df, G_pub, pub_id, id_pub, label_to_index,
                                          auth_plot_dump_dir,
                                          plot=plot)
            label_author_pub_dict = run_pagerank(probs, df, G_auth_pub, author_pub_id, id_author_pub,
                                                 label_to_index,
                                                 auth_plot_dump_dir,
                                                 plot=plot)
            label_phrase_dict, label_author_dict, label_pub_dict, label_author_pub_dict = rank_phrase_author_attr_author_attr_together(
                label_phrase_dict,
                label_author_dict,
                label_pub_dict,
                label_author_pub_dict,
                phrase_docid_map,
                author_docid_map,
                pub_docid_map,
                author_pub_docid_map,
                df,
                labels, i)

        # RANKING WITH ITERATION
        # label_phrase_dict, label_author_dict = rank_phrase_author_with_iteration(label_phrase_dict, label_author_dict,
        #                                                                          df, pred_labels, i)

        if plot:
            phrase_count = get_entity_count(label_phrase_dict, phrase_count)
            author_count = get_entity_count(label_author_dict, author_count)
            phrase_ppr_cutoff = get_cut_off(label_phrase_dict, phrase_ppr_cutoff)
            author_ppr_cutoff = get_cut_off(label_author_dict, author_ppr_cutoff)

        if should_print:
            print_label_phrase_dict(label_phrase_dict, id_phrase_map)
            print_label_entity_dict(label_author_dict)
            print_label_entity_dict(label_pub_dict)
            print_label_entity_dict(label_author_pub_dict)
        print("#" * 80)

    if plot:
        phrase_count_plot_dump_dir = pkl_dump_dir + "images/" + model_name + "/phrase/count_cutoff/"
        auth_count_plot_dump_dir = pkl_dump_dir + "images/" + model_name + "/author/count_cutoff/"
        os.makedirs(phrase_count_plot_dump_dir, exist_ok=True)
        os.makedirs(auth_count_plot_dump_dir, exist_ok=True)

        for l in phrase_count:
            y_values = phrase_count[l]
            x_values = range(1, t + 1)
            path = phrase_count_plot_dump_dir + l + "_phrase_count_iterations.png"
            x_label = "Iteration"
            y_label = "Number of Phrases"
            plot_entity_count(y_values, x_values, path, x_label, y_label)

        for l in author_count:
            y_values = author_count[l]
            x_values = range(1, t + 1)
            path = auth_count_plot_dump_dir + l + "_author_count_iterations.png"
            x_label = "Iteration"
            y_label = "Number of Authors"
            plot_entity_count(y_values, x_values, path, x_label, y_label)

        for l in phrase_ppr_cutoff:
            y_values = phrase_ppr_cutoff[l]
            x_values = range(1, t + 1)
            path = phrase_count_plot_dump_dir + l + "_phrase_ppr_cutoff_iterations.png"
            x_label = "Iteration"
            y_label = "PPR Cutoff"
            plot_entity_count(y_values, x_values, path, x_label, y_label)

        for l in author_ppr_cutoff:
            y_values = author_ppr_cutoff[l]
            x_values = range(1, t + 1)
            path = auth_count_plot_dump_dir + l + "_author_ppr_cutoff_iterations.png"
            x_label = "Iteration"
            y_label = "PPR Cutoff"
            plot_entity_count(y_values, x_values, path, x_label, y_label)