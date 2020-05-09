from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pickle
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)


def get_label_w2v_dict(label_term_dict):
    label_w2v_dict = {}
    for l in label_term_dict:
        temp = np.zeros((100,))
        for w in label_term_dict[l]:
            try:
                temp += model[w]
            except Exception as e:
                print("Word ", w, e)
        label_w2v_dict[l] = temp
    return label_w2v_dict


def get_label_term_json(path):
    import json
    dic = json.load(open(path, "r"))
    return dic


if __name__ == "__main__":
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "dblp/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_mapped_labels_phrase_removed_stopwords.pkl", "rb") as handler:
        df = pickle.load(handler)

    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords_run3.json")

    tagged_data = [word_tokenize(_d) for i, _d in enumerate(df["sentence"])]
    max_epochs = 20
    vec_size = 100
    alpha = 0.025

    model = Word2Vec(tagged_data, size=vec_size, alpha=alpha, min_count=1)
    # model.save(pkl_dump_dir + "w2v.model_parent")
    print("Model Saved")

    # model = Word2Vec.load(pkl_dump_dir + "w2v.model")

    label_w2v_dict = get_label_w2v_dict(label_term_dict)

    pred = []

    for i, row in df.iterrows():
        words = word_tokenize(row["sentence"].lower())
        temp = np.zeros((100,))
        for w in words:
            try:
                temp += model[w]
            except Exception as e:
                pass
                # print("Word: ", w, e)
        maxi = -1
        max_l = ""
        for l in label_w2v_dict:
            cos = cosine_similarity((temp / len(words)).reshape(1, -1), label_w2v_dict[l].reshape(1, -1))[0][0]
            if cos > maxi:
                maxi = cos
                max_l = l
        pred.append(max_l)

    print(classification_report(df["label"], pred))
