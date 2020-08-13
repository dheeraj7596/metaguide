from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from scipy.special import softmax
from keras.losses import kullback_leibler_divergence
import matplotlib.pyplot as plt
from data_utils import *
from analyze_utils import analyze
from bert_train import train_bert, test
from cnn_model.train_cnn import train_cnn
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def create_training_df(X, y, y_true):
    dic = {}
    dic["Data"] = X
    dic["Training label"] = y
    dic["True label"] = y_true
    df_X = DataFrame(dic)
    return df_X


def get_distinct_labels(df):
    label_to_index = {}
    index_to_label = {}
    labels = set(df["label"])

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def get_entity_count(label_entity_dict, entity_count):
    for l in label_entity_dict:
        try:
            entity_count[l].append(len(label_entity_dict[l]))
        except:
            entity_count[l] = [len(label_entity_dict[l])]
    return entity_count


def get_cut_off(label_entity_dict, entity_cut_off):
    for l in label_entity_dict:
        items_list = list(label_entity_dict[l].items())
        try:
            entity_cut_off[l].append(items_list[-1][1])
        except:
            entity_cut_off[l] = [items_list[-1][1]]
    return entity_cut_off


def plot_entity_count(y_values, x_values, path, x_label, y_label):
    plt.figure()
    plt.plot(x_values, y_values)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(path)


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


def get_train_data(df, labels, label_term_dict, label_author_dict, label_pub_dict, label_year_dict,
                   label_author_pub_dict, label_pub_year_dict, label_author_year_dict, tokenizer, label_to_index,
                   soft=False, clf="HAN"):
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
        authors_set = set(row["authors"])
        pub = row["publisher"]
        line = row["text"]
        label = row["label"]
        year = row["publication_year"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        l_phrase = get_phrase_label(words, label_term_dict, labels)
        l_metadata = get_metadata_label(authors_set, label_author_dict, label_pub_dict, label_year_dict,
                                        label_author_pub_dict, label_pub_year_dict, label_author_year_dict, row, labels)
        y_phrase.append(l_phrase)
        y_metadata.append(l_metadata)
        for l in labels:
            seed_words = set(label_term_dict[l].keys())
            int_labels = list(set(words).intersection(seed_words))

            if len(label_author_dict) > 0:
                seed_authors = set(label_author_dict[l].keys())
                int_authors = authors_set.intersection(seed_authors)
            else:
                int_authors = []

            if len(label_pub_dict) and len(label_pub_dict[l]) > 0:
                seed_pubs = set(label_pub_dict[l].keys())
                if pub in seed_pubs:
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    count_dict[l]["PUB_" + str(pub)] = label_pub_dict[l][pub]
                    flag = 1

            if len(label_year_dict) and len(label_year_dict[l]) > 0:
                seed_years = set(label_year_dict[l].keys())
                if year in seed_years:
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    count_dict[l]["YEAR_" + str(year)] = label_year_dict[l][year]
                    flag = 1

            if len(label_pub_year_dict) and len(label_pub_year_dict[l]) > 0:
                seed_pub_years = set(label_pub_year_dict[l].keys())
                if (pub, year) in seed_pub_years:
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    count_dict[l]["PUB_YEAR_" + str((pub, year))] = label_pub_year_dict[l][(pub, year)]
                    flag = 1

            if len(label_author_pub_dict) > 0 and len(label_author_pub_dict[l]):
                seed_author_pubs = set(label_author_pub_dict[l].keys())
                row_auth_pubs = set()
                for auth in authors_set:
                    row_auth_pubs.add((auth, pub))
                int_auth_pubs = row_auth_pubs.intersection(seed_author_pubs)
            else:
                int_auth_pubs = []

            if len(label_author_year_dict) > 0 and len(label_author_year_dict[l]):
                seed_author_years = set(label_author_year_dict[l].keys())
                row_auth_years = set()
                for auth in authors_set:
                    row_auth_years.add((auth, year))
                int_auth_years = row_auth_years.intersection(seed_author_years)
            else:
                int_auth_years = []

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

            for auth in int_authors:
                try:
                    temp = count_dict[l]
                except:
                    count_dict[l] = {}
                count_dict[l]["AUTH_" + str(auth)] = label_author_dict[l][auth]
                flag = 1

            for auth_pub in int_auth_pubs:
                try:
                    temp = count_dict[l]
                except:
                    count_dict[l] = {}
                count_dict[l]["AUTH_PUB_" + str(auth_pub)] = label_author_pub_dict[l][auth_pub]
                flag = 1

            for auth_year in int_auth_years:
                try:
                    temp = count_dict[l]
                except:
                    count_dict[l] = {}
                count_dict[l]["AUTH_YEAR_" + str(auth_year)] = label_author_year_dict[l][auth_year]
                flag = 1

        if flag:
            if not soft:
                lbl = argmax_label(count_dict)
                if not lbl:
                    continue
            else:
                lbl = softmax_label(count_dict, label_to_index)
            y.append(lbl)
            if clf == "BERT":
                X.append(index)
            else:
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


def get_metadata_label(authors_set, label_author_dict, label_pub_dict, label_year_dict, label_author_pub_dict,
                       label_pub_year_dict, label_author_year_dict, row, labels):
    count_dict = {}
    flag = 0
    pub = row["publisher"]
    year = row["publication_year"]
    for l in labels:
        if len(label_author_dict) > 0:
            seed_authors = set(label_author_dict[l].keys())
            int_authors = authors_set.intersection(seed_authors)
        else:
            int_authors = []

        if len(label_pub_dict) > 0:
            seed_pubs = set(label_pub_dict[l].keys())
            int_pubs = {pub}.intersection(seed_pubs)
        else:
            int_pubs = []

        if len(label_year_dict) > 0:
            seed_years = set(label_year_dict[l].keys())
            int_years = {year}.intersection(seed_years)
        else:
            int_years = []

        if len(label_pub_year_dict) > 0:
            seed_pub_years = set(label_pub_year_dict[l].keys())
            int_pub_years = {(pub, year)}.intersection(seed_pub_years)
        else:
            int_pub_years = []

        if len(label_author_pub_dict) > 0:
            seed_author_pubs = set(label_author_pub_dict[l].keys())
            row_auth_pubs = set()
            for auth in authors_set:
                row_auth_pubs.add((auth, pub))
            int_auth_pubs = row_auth_pubs.intersection(seed_author_pubs)
        else:
            int_auth_pubs = []

        if len(label_author_year_dict) > 0:
            seed_author_years = set(label_author_year_dict[l].keys())
            row_auth_years = set()
            for auth in authors_set:
                row_auth_years.add((auth, year))
            int_auth_years = row_auth_years.intersection(seed_author_years)
        else:
            int_auth_years = []

        for auth in int_authors:
            flag = 1
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["AUTH_" + str(auth)] = label_author_dict[l][auth]

        for pub in int_pubs:
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["PUB_" + str(pub)] = label_pub_dict[l][pub]
            flag = 1

        for y in int_years:
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["YEAR_" + str(y)] = label_year_dict[l][y]
            flag = 1

        for auth_pub in int_auth_pubs:
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["AUTH_PUB_" + str(auth_pub)] = label_author_pub_dict[l][auth_pub]
            flag = 1

        for auth_year in int_auth_years:
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["AUTH_YEAR_" + str(auth_year)] = label_author_year_dict[l][auth_year]
            flag = 1

        for pub_year in int_pub_years:
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["PUB_YEAR_" + str(pub_year)] = label_pub_year_dict[l][pub_year]
            flag = 1

    lbl = None
    if flag:
        lbl = argmax_label(count_dict)
    return lbl


def get_confident_train_data(df, labels, label_term_dict, label_author_dict, label_pub_dict, label_year_dict,
                             label_author_pub_dict, label_pub_year_dict, label_author_year_dict, tokenizer):
    y = []
    y_phrase = []
    y_metadata = []
    X = []
    y_true = []
    y_true_phrase = []
    y_true_metadata = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        authors_set = set(row["authors"])
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])

        l_phrase = get_phrase_label(words, label_term_dict, labels)
        l_metadata = get_metadata_label(authors_set, label_author_dict, label_pub_dict, label_year_dict,
                                        label_author_pub_dict, label_pub_year_dict, label_author_year_dict, row, labels)

        if l_phrase == l_metadata:
            y.append(l_phrase)
            X.append(line)
            y_true.append(label)
        elif l_phrase is None:
            y.append(l_metadata)
            X.append(line)
            y_true.append(label)
        elif l_metadata is None:
            y.append(l_phrase)
            X.append(line)
            y_true.append(label)

        if l_phrase is not None:
            y_phrase.append(l_phrase)
            y_true_phrase.append(label)
        if l_metadata is not None:
            y_metadata.append(l_metadata)
            y_true_metadata.append(label)

    print("****************** CLASSIFICATION REPORT FOR PHRASE LABELS ********************")
    print(classification_report(y_true_phrase, y_phrase))

    print("****************** CLASSIFICATION REPORT FOR METADATA LABELS ********************")
    print(classification_report(y_true_metadata, y_metadata))
    return X, y, y_true


def train_classifier(df, labels, label_term_dict, label_author_dict, label_pub_dict, label_year_dict,
                     label_author_pub_dict, label_pub_year_dict, label_author_year_dict, label_to_index, index_to_label,
                     model_name, clf, use_gpu, old=True, soft=False):
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "books/"
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

    if old:
        X, y, y_true = get_train_data(df, labels, label_term_dict, label_author_dict, label_pub_dict, label_year_dict,
                                      label_author_pub_dict, label_pub_year_dict, label_author_year_dict, tokenizer,
                                      label_to_index, soft=soft, clf=clf)
        if clf == "BERT":
            df_orig = pickle.load(open(basepath + dataset + "df.pkl", "rb"))
            X = list(df_orig.iloc[X]["text"])
    else:
        X, y, y_true = get_confident_train_data(df, labels, label_term_dict, label_author_dict, label_pub_dict,
                                                label_year_dict, label_author_pub_dict, label_pub_year_dict,
                                                label_author_year_dict, tokenizer)
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

    # print("Creating Embedding matrix...")
    # embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)
    if clf == "HAN":
        print("Splitting into train, dev...")
        X_train, y_train, X_val, y_val, _, _ = create_train_dev(X, labels=y_vec, tokenizer=tokenizer,
                                                                max_sentences=max_sentences,
                                                                max_sentence_length=max_sentence_length,
                                                                max_words=max_words, val=False)
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
        print("Dumping the model...")
        model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
        model.save(dump_dir + "model_" + model_name + ".h5")
    elif clf == "BERT":
        y_vec = []
        for lbl_ in y:
            y_vec.append(label_to_index[lbl_])
        model = train_bert(X, y_vec, use_gpu)

        y_true_all = []
        for lbl_ in df.label:
            y_true_all.append(label_to_index[lbl_])

        predictions = test(model, df_orig["text"], y_true_all, use_gpu)
        for i, p in enumerate(predictions):
            if i == 0:
                pred = p
            else:
                pred = np.concatenate((pred, p))

        pred_labels = []
        for p in pred:
            pred_labels.append(index_to_label[p.argmax(axis=-1)])
        y_true_all = df["label"]
    elif clf == "CNN":
        y_vec = []
        for lbl_ in y:
            y_vec.append(label_to_index[lbl_])

        y_true_all = []
        for lbl_ in df.label:
            y_true_all.append(label_to_index[lbl_])

        pred_idxs, pred, true_idxs = train_cnn(X, y_vec, df["text"], y_true_all, use_gpu)

        pred_labels = []
        for p in pred_idxs:
            pred_labels.append(index_to_label[p])

        y_true_all = []
        for p in true_idxs:
            y_true_all.append(index_to_label[p])
    else:
        raise ValueError("clf can only be HAN or BERT or CNN")
    print(classification_report(y_true_all, pred_labels))
    return pred_labels, pred
