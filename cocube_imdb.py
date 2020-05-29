from cocube_utils_beta_imdb import get_distinct_labels, train_classifier
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
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset
    model_name = sys.argv[2]
    is_soft = int(sys.argv[3])

    df = pickle.load(
        open(pkl_dump_dir + "df_summary_top6_title_summary_all_reviews_removed_stopwords_metadata.pkl", "rb"))
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
    G_adult = sparse.load_npz(pkl_dump_dir + "G_adult.npz")
    G_actor = sparse.load_npz(pkl_dump_dir + "G_actor.npz")
    G_actress = sparse.load_npz(pkl_dump_dir + "G_actress.npz")
    G_producer = sparse.load_npz(pkl_dump_dir + "G_producer.npz")
    G_writer = sparse.load_npz(pkl_dump_dir + "G_writer.npz")
    G_director = sparse.load_npz(pkl_dump_dir + "G_director.npz")
    G_composer = sparse.load_npz(pkl_dump_dir + "G_composer.npz")
    G_cinematographer = sparse.load_npz(pkl_dump_dir + "G_cinematographer.npz")
    G_editor = sparse.load_npz(pkl_dump_dir + "G_editor.npz")
    G_prod_designer = sparse.load_npz(pkl_dump_dir + "G_prod_designer.npz")

    G_dir_adult = sparse.load_npz(pkl_dump_dir + "G_dir_adult.npz")
    G_dir_actor = sparse.load_npz(pkl_dump_dir + "G_dir_actor.npz")
    G_dir_actress = sparse.load_npz(pkl_dump_dir + "G_dir_actress.npz")
    G_dir_producer = sparse.load_npz(pkl_dump_dir + "G_dir_producer.npz")
    G_dir_writer = sparse.load_npz(pkl_dump_dir + "G_dir_writer.npz")
    G_dir_composer = sparse.load_npz(pkl_dump_dir + "G_dir_composer.npz")
    G_dir_cinematographer = sparse.load_npz(pkl_dump_dir + "G_dir_cinematographer.npz")
    G_dir_editor = sparse.load_npz(pkl_dump_dir + "G_dir_editor.npz")
    G_dir_prod_designer = sparse.load_npz(pkl_dump_dir + "G_dir_prod_designer.npz")
    G_actor_actress = sparse.load_npz(pkl_dump_dir + "G_actor_actress.npz")

    fnust_id = pickle.load(open(pkl_dump_dir + "fnust_id.pkl", "rb"))
    id_fnust = pickle.load(open(pkl_dump_dir + "id_fnust.pkl", "rb"))

    adult_id = pickle.load(open(pkl_dump_dir + "adult_id.pkl", "rb"))
    id_adult = pickle.load(open(pkl_dump_dir + "id_adult.pkl", "rb"))

    actor_id = pickle.load(open(pkl_dump_dir + "actor_id.pkl", "rb"))
    id_actor = pickle.load(open(pkl_dump_dir + "id_actor.pkl", "rb"))

    actress_id = pickle.load(open(pkl_dump_dir + "actress_id.pkl", "rb"))
    id_actress = pickle.load(open(pkl_dump_dir + "id_actress.pkl", "rb"))

    producer_id = pickle.load(open(pkl_dump_dir + "producer_id.pkl", "rb"))
    id_producer = pickle.load(open(pkl_dump_dir + "id_producer.pkl", "rb"))

    writer_id = pickle.load(open(pkl_dump_dir + "writer_id.pkl", "rb"))
    id_writer = pickle.load(open(pkl_dump_dir + "id_writer.pkl", "rb"))

    director_id = pickle.load(open(pkl_dump_dir + "director_id.pkl", "rb"))
    id_director = pickle.load(open(pkl_dump_dir + "id_director.pkl", "rb"))

    composer_id = pickle.load(open(pkl_dump_dir + "composer_id.pkl", "rb"))
    id_composer = pickle.load(open(pkl_dump_dir + "id_composer.pkl", "rb"))

    cinematographer_id = pickle.load(open(pkl_dump_dir + "cinematographer_id.pkl", "rb"))
    id_cinematographer = pickle.load(open(pkl_dump_dir + "id_cinematographer.pkl", "rb"))

    editor_id = pickle.load(open(pkl_dump_dir + "editor_id.pkl", "rb"))
    id_editor = pickle.load(open(pkl_dump_dir + "id_editor.pkl", "rb"))

    prod_designer_id = pickle.load(open(pkl_dump_dir + "prod_designer_id.pkl", "rb"))
    id_prod_designer = pickle.load(open(pkl_dump_dir + "id_prod_designer.pkl", "rb"))

    dir_adult_id = pickle.load(open(pkl_dump_dir + "dir_adult_id.pkl", "rb"))
    id_dir_adult = pickle.load(open(pkl_dump_dir + "id_dir_adult.pkl", "rb"))

    dir_actor_id = pickle.load(open(pkl_dump_dir + "dir_actor_id.pkl", "rb"))
    id_dir_actor = pickle.load(open(pkl_dump_dir + "id_dir_actor.pkl", "rb"))

    dir_actress_id = pickle.load(open(pkl_dump_dir + "dir_actress_id.pkl", "rb"))
    id_dir_actress = pickle.load(open(pkl_dump_dir + "id_dir_actress.pkl", "rb"))

    dir_producer_id = pickle.load(open(pkl_dump_dir + "dir_producer_id.pkl", "rb"))
    id_dir_producer = pickle.load(open(pkl_dump_dir + "id_dir_producer.pkl", "rb"))

    dir_writer_id = pickle.load(open(pkl_dump_dir + "dir_writer_id.pkl", "rb"))
    id_dir_writer = pickle.load(open(pkl_dump_dir + "id_dir_writer.pkl", "rb"))

    dir_composer_id = pickle.load(open(pkl_dump_dir + "dir_composer_id.pkl", "rb"))
    id_dir_composer = pickle.load(open(pkl_dump_dir + "id_dir_composer.pkl", "rb"))

    dir_cinematographer_id = pickle.load(open(pkl_dump_dir + "dir_cinematographer_id.pkl", "rb"))
    id_dir_cinematographer = pickle.load(open(pkl_dump_dir + "id_dir_cinematographer.pkl", "rb"))

    dir_editor_id = pickle.load(open(pkl_dump_dir + "dir_editor_id.pkl", "rb"))
    id_dir_editor = pickle.load(open(pkl_dump_dir + "id_dir_editor.pkl", "rb"))

    dir_prod_designer_id = pickle.load(open(pkl_dump_dir + "dir_prod_designer_id.pkl", "rb"))
    id_dir_prod_designer = pickle.load(open(pkl_dump_dir + "id_dir_prod_designer.pkl", "rb"))

    actor_actress_id = pickle.load(open(pkl_dump_dir + "actor_actress_id.pkl", "rb"))
    id_actor_actress = pickle.load(open(pkl_dump_dir + "id_actor_actress.pkl", "rb"))

    # Loading Docid maps
    phrase_docid_map = pickle.load(open(pkl_dump_dir + "phrase_docid_map.pkl", "rb"))
    adult_docid_map = pickle.load(open(pkl_dump_dir + "adult_docid_map.pkl", "rb"))
    actor_docid_map = pickle.load(open(pkl_dump_dir + "actor_docid_map.pkl", "rb"))
    actress_docid_map = pickle.load(open(pkl_dump_dir + "actress_docid_map.pkl", "rb"))
    producer_docid_map = pickle.load(open(pkl_dump_dir + "producer_docid_map.pkl", "rb"))
    writer_docid_map = pickle.load(open(pkl_dump_dir + "writer_docid_map.pkl", "rb"))
    director_docid_map = pickle.load(open(pkl_dump_dir + "director_docid_map.pkl", "rb"))
    composer_docid_map = pickle.load(open(pkl_dump_dir + "composer_docid_map.pkl", "rb"))
    cinematographer_docid_map = pickle.load(open(pkl_dump_dir + "cinematographer_docid_map.pkl", "rb"))
    editor_docid_map = pickle.load(open(pkl_dump_dir + "editor_docid_map.pkl", "rb"))
    prod_designer_docid_map = pickle.load(open(pkl_dump_dir + "prod_designer_docid_map.pkl", "rb"))
    dir_adult_docid_map = pickle.load(open(pkl_dump_dir + "dir_adult_docid_map.pkl", "rb"))
    dir_actor_docid_map = pickle.load(open(pkl_dump_dir + "dir_actor_docid_map.pkl", "rb"))
    dir_actress_docid_map = pickle.load(open(pkl_dump_dir + "dir_actress_docid_map.pkl", "rb"))
    dir_producer_docid_map = pickle.load(open(pkl_dump_dir + "dir_producer_docid_map.pkl", "rb"))
    dir_writer_docid_map = pickle.load(open(pkl_dump_dir + "dir_writer_docid_map.pkl", "rb"))
    dir_composer_docid_map = pickle.load(open(pkl_dump_dir + "dir_composer_docid_map.pkl", "rb"))
    dir_cinematographer_docid_map = pickle.load(open(pkl_dump_dir + "dir_cinematographer_docid_map.pkl", "rb"))
    dir_editor_docid_map = pickle.load(open(pkl_dump_dir + "dir_editor_docid_map.pkl", "rb"))
    dir_prod_designer_docid_map = pickle.load(open(pkl_dump_dir + "dir_prod_designer_docid_map.pkl", "rb"))
    actor_actress_docid_map = pickle.load(open(pkl_dump_dir + "actor_actress_docid_map.pkl", "rb"))

    label_phrase_dict = modify(label_term_dict)
    print_label_phrase_dict(label_phrase_dict, id_phrase_map)
    label_adult_dict = {}
    label_actor_dict = {}
    label_actress_dict = {}
    label_producer_dict = {}
    label_writer_dict = {}
    label_director_dict = {}
    label_composer_dict = {}
    label_cinematographer_dict = {}
    label_editor_dict = {}
    label_prod_designer_dict = {}
    label_dir_adult_dict = {}
    label_dir_actor_dict = {}
    label_dir_actress_dict = {}
    label_dir_producer_dict = {}
    label_dir_writer_dict = {}
    label_dir_composer_dict = {}
    label_dir_cinematographer_dict = {}
    label_dir_editor_dict = {}
    label_dir_prod_designer_dict = {}
    label_actor_actress_dict = {}

    t = 9
    pre_train = 0
    plot = False
    should_print = True
    algo = int(sys.argv[1])

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
            pred_labels, probs = train_classifier(df, labels, label_phrase_dict,
                                                  label_adult_dict,
                                                  label_actor_dict,
                                                  label_actress_dict,
                                                  label_producer_dict,
                                                  label_writer_dict,
                                                  label_director_dict,
                                                  label_composer_dict,
                                                  label_cinematographer_dict,
                                                  label_editor_dict,
                                                  label_prod_designer_dict,
                                                  label_dir_adult_dict,
                                                  label_dir_actor_dict,
                                                  label_dir_actress_dict,
                                                  label_dir_producer_dict,
                                                  label_dir_writer_dict,
                                                  label_dir_composer_dict,
                                                  label_dir_cinematographer_dict,
                                                  label_dir_editor_dict,
                                                  label_dir_prod_designer_dict,
                                                  label_actor_actress_dict, label_to_index, index_to_label, model_name,
                                                  soft=is_soft)

        # RANKING PHRASE ONLY
        if algo == 1:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             None,
                                             plot=plot)
            label_phrase_dict = rank_phrase_only(label_phrase_dict, phrase_docid_map, df, labels, i)

        # RANKING PHRASE, METADATA TOGETHER
        elif algo == 2:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             None,
                                             plot=plot)
            label_adult_dict = run_pagerank(probs, df, G_adult, adult_id, id_adult, label_to_index,
                                            None,
                                            plot=plot)
            label_actor_dict = run_pagerank(probs, df, G_actor, actor_id, id_actor, label_to_index,
                                            None,
                                            plot=plot)
            label_actress_dict = run_pagerank(probs, df, G_actress, actress_id, id_actress, label_to_index,
                                              None,
                                              plot=plot)
            label_producer_dict = run_pagerank(probs, df, G_producer, producer_id, id_producer, label_to_index,
                                               None,
                                               plot=plot)
            label_writer_dict = run_pagerank(probs, df, G_writer, writer_id, id_writer, label_to_index,
                                             None,
                                             plot=plot)
            label_director_dict = run_pagerank(probs, df, G_director, director_id, id_director, label_to_index,
                                               None,
                                               plot=plot)
            label_composer_dict = run_pagerank(probs, df, G_composer, composer_id, id_composer, label_to_index,
                                               None,
                                               plot=plot)
            label_cinematographer_dict = run_pagerank(probs, df, G_cinematographer, cinematographer_id,
                                                      id_cinematographer, label_to_index,
                                                      None,
                                                      plot=plot)
            label_editor_dict = run_pagerank(probs, df, G_editor, editor_id, id_editor, label_to_index,
                                             None,
                                             plot=plot)
            label_prod_designer_dict = run_pagerank(probs, df, G_prod_designer, prod_designer_id, id_prod_designer,
                                                    label_to_index,
                                                    None,
                                                    plot=plot)
            label_dir_adult_dict = run_pagerank(probs, df, G_dir_adult, dir_adult_id, id_dir_adult, label_to_index,
                                                None,
                                                plot=plot)
            label_dir_actor_dict = run_pagerank(probs, df, G_dir_actor, dir_actor_id, id_dir_actor, label_to_index,
                                                None,
                                                plot=plot)
            label_dir_actress_dict = run_pagerank(probs, df, G_dir_actress, dir_actress_id, id_dir_actress,
                                                  label_to_index,
                                                  None,
                                                  plot=plot)
            label_dir_producer_dict = run_pagerank(probs, df, G_dir_producer, dir_producer_id, id_dir_producer,
                                                   label_to_index,
                                                   None,
                                                   plot=plot)
            label_dir_writer_dict = run_pagerank(probs, df, G_dir_writer, dir_writer_id, id_dir_writer, label_to_index,
                                                 None,
                                                 plot=plot)
            label_dir_composer_dict = run_pagerank(probs, df, G_dir_composer, dir_composer_id, id_dir_composer,
                                                   label_to_index,
                                                   None,
                                                   plot=plot)
            label_dir_cinematographer_dict = run_pagerank(probs, df, G_dir_cinematographer, dir_cinematographer_id,
                                                          id_dir_cinematographer, label_to_index,
                                                          None,
                                                          plot=plot)
            label_dir_editor_dict = run_pagerank(probs, df, G_dir_editor, dir_editor_id, id_dir_editor, label_to_index,
                                                 None,
                                                 plot=plot)
            label_dir_prod_designer_dict = run_pagerank(probs, df, G_dir_prod_designer, dir_prod_designer_id,
                                                        id_dir_prod_designer, label_to_index,
                                                        None,
                                                        plot=plot)
            label_actor_actress_dict = run_pagerank(probs, df, G_actor_actress, actor_actress_id, id_actor_actress,
                                                    label_to_index,
                                                    None,
                                                    plot=plot)

            label_entity_dict_list = [label_phrase_dict,
                                      label_adult_dict,
                                      label_actor_dict,
                                      label_actress_dict,
                                      label_producer_dict,
                                      label_writer_dict,
                                      label_director_dict,
                                      label_composer_dict,
                                      label_cinematographer_dict,
                                      label_editor_dict,
                                      label_prod_designer_dict,
                                      label_dir_adult_dict,
                                      label_dir_actor_dict,
                                      label_dir_actress_dict,
                                      label_dir_producer_dict,
                                      label_dir_writer_dict,
                                      label_dir_composer_dict,
                                      label_dir_cinematographer_dict,
                                      label_dir_editor_dict,
                                      label_dir_prod_designer_dict,
                                      label_actor_actress_dict]

            entity_docid_map_list = [phrase_docid_map,
                                     adult_docid_map,
                                     actor_docid_map,
                                     actress_docid_map,
                                     producer_docid_map,
                                     writer_docid_map,
                                     director_docid_map,
                                     composer_docid_map,
                                     cinematographer_docid_map,
                                     editor_docid_map,
                                     prod_designer_docid_map,
                                     dir_adult_docid_map,
                                     dir_actor_docid_map,
                                     dir_actress_docid_map,
                                     dir_producer_docid_map,
                                     dir_writer_docid_map,
                                     dir_composer_docid_map,
                                     dir_cinematographer_docid_map,
                                     dir_editor_docid_map,
                                     dir_prod_designer_docid_map,
                                     actor_actress_docid_map]
            label_phrase_dict, label_adult_dict, label_actor_dict, label_actress_dict, label_producer_dict, label_writer_dict, label_director_dict, label_composer_dict, label_cinematographer_dict, label_editor_dict, label_prod_designer_dict, label_dir_adult_dict, label_dir_actor_dict, label_dir_actress_dict, label_dir_producer_dict, label_dir_writer_dict, label_dir_composer_dict, label_dir_cinematographer_dict, label_dir_editor_dict, label_dir_prod_designer_dict, label_actor_actress_dict = rank_phrase_metadata_together(
                label_entity_dict_list, entity_docid_map_list, df, labels, i)

        elif algo == 3:
            label_phrase_dict = run_pagerank(probs, df, G_phrase, fnust_id, id_fnust, label_to_index,
                                             None,
                                             plot=plot)

            label_composer_dict = run_pagerank(probs, df, G_composer, composer_id, id_composer, label_to_index,
                                               None,
                                               plot=plot)

            label_dir_adult_dict = run_pagerank(probs, df, G_dir_adult, dir_adult_id, id_dir_adult, label_to_index,
                                                None,
                                                plot=plot)

            label_dir_actor_dict = run_pagerank(probs, df, G_dir_actor, dir_actor_id, id_dir_actor, label_to_index,
                                                None,
                                                plot=plot)

            label_dir_composer_dict = run_pagerank(probs, df, G_dir_composer, dir_composer_id, id_dir_composer,
                                                   label_to_index,
                                                   None,
                                                   plot=plot)

            label_dir_producer_dict = run_pagerank(probs, df, G_dir_producer, dir_producer_id, id_dir_producer,
                                                   label_to_index,
                                                   None,
                                                   plot=plot)

            label_dir_editor_dict = run_pagerank(probs, df, G_dir_editor, dir_editor_id, id_dir_editor, label_to_index,
                                                 None,
                                                 plot=plot)

            label_dir_prod_designer_dict = run_pagerank(probs, df, G_dir_prod_designer, dir_prod_designer_id,
                                                        id_dir_prod_designer, label_to_index,
                                                        None,
                                                        plot=plot)

            label_actor_actress_dict = run_pagerank(probs, df, G_actor_actress, actor_actress_id, id_actor_actress,
                                                    label_to_index,
                                                    None,
                                                    plot=plot)

            label_entity_dict_list = [label_phrase_dict,
                                      label_composer_dict,
                                      label_dir_adult_dict,
                                      label_dir_actor_dict,
                                      label_dir_composer_dict,
                                      label_dir_producer_dict,
                                      label_dir_editor_dict,
                                      label_dir_prod_designer_dict,
                                      label_actor_actress_dict]

            entity_docid_map_list = [phrase_docid_map,
                                     composer_docid_map,
                                     dir_adult_docid_map,
                                     dir_actor_docid_map,
                                     dir_composer_docid_map,
                                     dir_producer_docid_map,
                                     dir_editor_docid_map,
                                     dir_prod_designer_docid_map,
                                     actor_actress_docid_map]
            label_phrase_dict, label_composer_dict, label_dir_adult_dict, label_dir_actor_dict, label_dir_composer_dict, label_dir_producer_dict, label_dir_editor_dict, label_dir_prod_designer_dict, label_actor_actress_dict = rank_phrase_metadata_together(
                label_entity_dict_list, entity_docid_map_list, df, labels, i)

        # RANKING WITH ITERATION
        # label_phrase_dict, label_author_dict = rank_phrase_author_with_iteration(label_phrase_dict, label_author_dict,
        #                                                                          df, pred_labels, i)

        if should_print:
            print_label_phrase_dict(label_phrase_dict, id_phrase_map)
            print_label_entity_dict(label_adult_dict)
            print_label_entity_dict(label_actor_dict)
            print_label_entity_dict(label_actress_dict)
            print_label_entity_dict(label_producer_dict)
            print_label_entity_dict(label_writer_dict)
            print_label_entity_dict(label_director_dict)
            print_label_entity_dict(label_composer_dict)
            print_label_entity_dict(label_cinematographer_dict)
            print_label_entity_dict(label_editor_dict)
            print_label_entity_dict(label_prod_designer_dict)
            print_label_entity_dict(label_dir_adult_dict)
            print_label_entity_dict(label_dir_actor_dict)
            print_label_entity_dict(label_dir_actress_dict)
            print_label_entity_dict(label_dir_producer_dict)
            print_label_entity_dict(label_dir_writer_dict)
            print_label_entity_dict(label_dir_composer_dict)
            print_label_entity_dict(label_dir_cinematographer_dict)
            print_label_entity_dict(label_dir_editor_dict)
            print_label_entity_dict(label_dir_prod_designer_dict)
            print_label_entity_dict(label_actor_actress_dict)
        print("#" * 80)
