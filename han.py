from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from data_utils import *
from model import *
import pickle
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_distinct_labels(dataset):
    basepath = "/data4/dheeraj/metaguide/"
    label_to_index = pickle.load(open(basepath + dataset + "label_id.pkl", "rb"))
    index_to_label = pickle.load(open(basepath + dataset + "id_label.pkl", "rb"))
    labels = list(label_to_index.keys())
    return labels, label_to_index, index_to_label


def get_train_data(df):
    X = list(df["sentence"])
    y = list(df["label"])
    return X, y


def get_pickle_dumps(pkl_dump_dir):
    X_train = pickle.load(open(pkl_dump_dir + "X_train.pkl", "rb"))
    X_val = pickle.load(open(pkl_dump_dir + "X_val.pkl", "rb"))
    X_test = pickle.load(open(pkl_dump_dir + "X_test.pkl", "rb"))

    y_train = pickle.load(open(pkl_dump_dir + "y_train.pkl", "rb"))
    y_val = pickle.load(open(pkl_dump_dir + "y_val.pkl", "rb"))
    y_test = pickle.load(open(pkl_dump_dir + "y_test.pkl", "rb"))
    return X_train, y_train, X_test, y_test, X_val, y_val


if __name__ == "__main__":
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "arxiv_cs/"

    glove_dir = basepath + "glove.6B"

    model_name = "han_word2vec_topk_dict"
    dump_dir = basepath + "models/" + dataset + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + dataset + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 50000
    embedding_dim = 100
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(basepath + dataset + "df_cs_2014_filtered.pkl", "rb"))
    df = create_df(df)
    labels, label_to_index, index_to_label = get_distinct_labels(dataset)

    X, y = get_train_data(df)

    y_one_hot = make_one_hot(y, label_to_index)

    # print("Fitting tokenizer...")
    # tokenizer = fit_get_tokenizer(X, max_words)

    print("Getting tokenizer")
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer_topk_dict.pkl", "rb"))

    print("Splitting into train, dev...")
    X_train, y_train, X_test, y_test, X_val, y_val = get_pickle_dumps(pkl_dump_dir)
    # X_train, y_train, X_test, y_test, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
    #                                                                   max_sentences=max_sentences,
    #                                                                   max_sentence_length=max_sentence_length,
    #                                                                   max_words=max_words)

    # print("Creating Embedding matrix...")
    # embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)

    print("Getting Embedding matrix...")
    embedding_matrix = pickle.load(open(basepath + dataset + "embedding_matrix_topk_dict.pkl", "rb"))

    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)

    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])

    print("****************** CLASSIFICATION REPORT ON TEST DATA ********************")
    pred = model.predict(X_test)
    pred_labels = get_from_one_hot(pred, index_to_label)
    true_labels = get_from_one_hot(y_test, index_to_label)
    print(classification_report(true_labels, pred_labels))

    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
