from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from scipy.special import softmax
from keras.losses import kullback_leibler_divergence
from data_utils import *
from analyze_utils import analyze
from data.imdb.metadata_utils import *
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_distinct_labels(df):
    label_to_index = {}
    index_to_label = {}
    labels = set(df["label"])

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def argmax_label(count_dict):
    maxi = 0
    max_label = None
    for l in count_dict:
        count = 0
        for t in count_dict[l]:
            count += count_dict[l][t]
        if count > maxi:
            maxi = count
            max_label = l

    return max_label


def softmax_label(count_dict, label_to_index):
    temp = [0] * len(label_to_index)
    for l in count_dict:
        count = 0
        for t in count_dict[l]:
            count += count_dict[l][t]
        temp[label_to_index[l]] = count
    return softmax(temp)


def get_train_data(df, labels,
                   label_term_dict,
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
                   label_actor_actress_dict, tokenizer, label_to_index, soft=False):
    y = []
    X = []
    y_true = []
    y_phrase = []
    y_metadata = []
    y_true_all = []
    y_pseudo_all = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        l_phrase = get_phrase_label(words, label_term_dict, labels)
        l_metadata = get_metadata_label(label_adult_dict,
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
                                        label_actor_actress_dict, row, labels)
        y_phrase.append(l_phrase)
        y_metadata.append(l_metadata)
        for l in labels:
            seed_words = set(label_term_dict[l].keys())
            int_labels = list(set(words).intersection(seed_words))

            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += label_term_dict[l][word]
                    except:
                        count_dict[l][word] = label_term_dict[l][word]

            count_dict, flag = get_int_adults(flag, count_dict, l, label_adult_dict, row)
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_actor_dict, row, "actor")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_actress_dict, row, "actress")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_producer_dict, row, "producer")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_writer_dict, row, "writer")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_director_dict, row, "director")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_composer_dict, row, "composer")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_cinematographer_dict, row, "cinematographer")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_editor_dict, row, "editor")
            count_dict, flag = get_int_unigram(flag, count_dict, l, label_prod_designer_dict, row, "prod_designer")

            count_dict, flag = get_int_dir_adult(flag, count_dict, l, label_dir_adult_dict, row)
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_actor_dict, row, "director", "actor")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_actress_dict, row, "director", "actress")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_producer_dict, row, "director", "producer")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_writer_dict, row, "director", "writer")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_composer_dict, row, "director", "composer")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_cinematographer_dict, row, "director",
                                              "cinematographer")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_editor_dict, row, "director", "editor")
            count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_prod_designer_dict, row, "director",
                                              "prod_designer")
            count_dict = get_int_bigram(flag, count_dict, l, label_actor_actress_dict, row, "actor", "actress")

        if flag:
            if not soft:
                lbl = argmax_label(count_dict)
                if not lbl:
                    continue
            else:
                lbl = softmax_label(count_dict, label_to_index)
            y.append(lbl)
            X.append(line)
            y_true.append(label)
            y_pseudo_all.append(lbl)
            y_true_all.append(label)
        else:
            y_pseudo_all.append(None)
            y_true_all.append(label)
    analyze(y_pseudo_all, y_phrase, y_metadata, y_true_all)
    return X, y, y_true


def get_phrase_label(words, label_term_dict, labels):
    count_dict = {}
    flag = 0
    for l in labels:
        seed_words = set(label_term_dict[l].keys())
        int_labels = list(set(words).intersection(seed_words))
        for word in words:
            if word in int_labels:
                flag = 1
                try:
                    temp = count_dict[l]
                except:
                    count_dict[l] = {}
                try:
                    count_dict[l][word] += label_term_dict[l][word]
                except:
                    count_dict[l][word] = label_term_dict[l][word]
    lbl = None
    if flag:
        lbl = argmax_label(count_dict)
    return lbl


def get_metadata_label(label_adult_dict,
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
                       label_actor_actress_dict, row, labels):
    count_dict = {}
    flag = 0
    for l in labels:
        count_dict, flag = get_int_adults(flag, count_dict, l, label_adult_dict, row)
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_actor_dict, row, "actor")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_actress_dict, row, "actress")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_producer_dict, row, "producer")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_writer_dict, row, "writer")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_director_dict, row, "director")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_composer_dict, row, "composer")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_cinematographer_dict, row, "cinematographer")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_editor_dict, row, "editor")
        count_dict, flag = get_int_unigram(flag, count_dict, l, label_prod_designer_dict, row, "prod_designer")

        count_dict, flag = get_int_dir_adult(flag, count_dict, l, label_dir_adult_dict, row)
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_actor_dict, row, "director", "actor")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_actress_dict, row, "director", "actress")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_producer_dict, row, "director", "producer")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_writer_dict, row, "director", "writer")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_composer_dict, row, "director", "composer")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_cinematographer_dict, row, "director",
                                          "cinematographer")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_editor_dict, row, "director", "editor")
        count_dict, flag = get_int_bigram(flag, count_dict, l, label_dir_prod_designer_dict, row, "director",
                                          "prod_designer")
        count_dict = get_int_bigram(flag, count_dict, l, label_actor_actress_dict, row, "actor", "actress")

    lbl = None
    if flag:
        lbl = argmax_label(count_dict)
    return lbl


def train_classifier(df,
                     labels,
                     label_term_dict,
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
                     label_actor_actress_dict,
                     label_to_index, index_to_label, model_name, soft=False):
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "imdb/"
    # glove_dir = basepath + "glove.6B"
    dump_dir = basepath + "models/" + dataset + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + dataset + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    embedding_dim = 100
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))

    X, y, y_true = get_train_data(df,
                                  labels,
                                  label_term_dict,
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
                                  label_actor_actress_dict, tokenizer, label_to_index, soft=soft)
    print("****************** CLASSIFICATION REPORT FOR TRAINING DATA ********************")
    # df_train = create_training_df(X, y, y_true)
    # df_train.to_csv(basepath + dataset + "training_label.csv")
    if not soft:
        y_vec = make_one_hot(y, label_to_index)
        print(classification_report(y_true, y))
    else:
        y_vec = np.array(y)
        y_argmax = np.argmax(y, axis=-1)
        y_str = []
        for i in y_argmax:
            y_str.append(index_to_label[i])
        print(classification_report(y_true, y_str))
    # print("Fitting tokenizer...")
    # tokenizer = fit_get_tokenizer(X, max_words)
    print("Getting tokenizer")
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))

    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val, _, _ = create_train_dev(X, labels=y_vec, tokenizer=tokenizer,
                                                            max_sentences=max_sentences,
                                                            max_sentence_length=max_sentence_length,
                                                            max_words=max_words, val=False)
    # print("Creating Embedding matrix...")
    # embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)
    print("Getting Embedding matrix...")
    embedding_matrix = pickle.load(open(basepath + dataset + "embedding_matrix.pkl", "rb"))
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    if not soft:
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    else:
        model.compile(loss=kullback_leibler_divergence, optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])
    # print("****************** CLASSIFICATION REPORT FOR DOCUMENTS WITH LABEL WORDS ********************")
    # X_label_all = prep_data(texts=X, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
    #                         tokenizer=tokenizer)
    # pred = model.predict(X_label_all)
    # pred_labels = get_from_one_hot(pred, index_to_label)
    # print(classification_report(y_true, pred_labels))
    print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
    X_all = prep_data(texts=df["text"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    y_true_all = df["label"]
    pred = model.predict(X_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(y_true_all, pred_labels))
    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
    return pred_labels, pred
