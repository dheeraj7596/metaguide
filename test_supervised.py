import pickle
from cocube_utils import get_distinct_labels
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras_han.model import HAN
from model import *
from data_utils import *
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)


def run(df):
    glove_dir = basepath + "glove.6B"
    model_name = "books_sup"
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    embedding_dim = 100

    X = df["text"]
    y = df["label"]
    y_true = df["label"]

    labels, label_to_index, index_to_label = get_distinct_labels(df)
    y_one_hot = make_one_hot(y, label_to_index)
    # y = np.array(y)
    print("Fitting tokenizer...")
    tokenizer = fit_get_tokenizer(X, max_words)
    # print("Getting tokenizer")
    # tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))
    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val, X_test, y_test = create_train_dev_test(X, labels=y_one_hot, tokenizer=tokenizer,
                                                                           max_sentences=max_sentences,
                                                                           max_sentence_length=max_sentence_length,
                                                                           max_words=max_words)
    print("Creating Embedding matrix...")
    embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)
    # print("Getting Embedding matrix...")
    # embedding_matrix = pickle.load(open(basepath + dataset + "embedding_matrix.pkl", "rb"))
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es])
    print("****************** CLASSIFICATION REPORT ********************")
    pred = model.predict(X_test)
    true_labels = get_from_one_hot(y_test, index_to_label)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(true_labels, pred_labels))


if __name__ == "__main__":
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "books/"
    pkl_dump_dir = basepath + dataset

    # df = pickle.load(open(pkl_dump_dir + "df_mapped_labels_phrase_removed_stopwords_test.pkl", "rb"))
    # df = pickle.load(open(pkl_dump_dir + "business_1review_shortlisted_thresh_3.pkl", "rb"))
    df = pickle.load(open(pkl_dump_dir + "df.pkl", "rb"))
    print("RUNNING BOOKS")
    run(df)
    print("*" * 80)
