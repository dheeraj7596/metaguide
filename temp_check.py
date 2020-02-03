import numpy as np
from scipy import sparse
from fast_pagerank import pagerank
import pickle

if __name__ == "__main__":
    basepath = "./data/"
    dataset = "dblp/"
    data_path = basepath + dataset

    G_conf = sparse.load_npz(data_path + "G_conf.npz")
    df = pickle.load(open(data_path + "df_mapped_labels_phrase_removed_stopwords.pkl", "rb"))
    venue_id = pickle.load(open(data_path + "venue_id.pkl", "rb"))
    id_venue = pickle.load(open(data_path + "id_venue.pkl", "rb"))

    count = len(venue_id) + len(df)
    start = len(df)
    labels = list(set(df.label))
    categories = list(df.label)

    top_conf_map = {}
    for l in labels:
        print("Pagerank running for: ", l)
        personalized = np.zeros((count,))
        for i, cat in enumerate(categories):
            if cat == l:
                personalized[i] = 1
        pr = pagerank(G_conf, p=0.85, personalize=personalized)
        temp_list = list(pr)[start:]
        sorted_temp_list = sorted(temp_list, reverse=True)
        args = np.argsort(temp_list)[::-1]
        top_auths = []
        for i in args:
            top_auths.append(id_venue[start + i])
        top_conf_map[l] = top_auths

    pass
