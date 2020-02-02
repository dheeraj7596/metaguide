import pickle
import numpy as np
from scipy import sparse
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power
import matplotlib.pyplot as plt


def make_auth_pair_map(df):
    start = len(df)
    auth_num_map = {}
    num_auth_map = {}
    auth_pairs_set = set()
    for doc_id in range(len(df)):
        auth_pairs_set.update(set(df.iloc[doc_id]["author pairs"]))

    for pair in auth_pairs_set:
        auth_num_map[pair] = start
        num_auth_map[start] = pair
        start += 1
    return auth_num_map, num_auth_map, start


if __name__ == "__main__":
    base_path = "/data4/dheeraj/metaguide/"
    dataset = "arxiv_cs/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_cs_2014_filtered_authorpairs.pkl", "rb"))

    start = len(df)
    auth_num_map, num_auth_map, count = make_auth_pair_map(df)

    edges = []
    weights = []
    for i, row in df.iterrows():
        for pair in row["author pairs"]:
            edges.append([i, auth_num_map[pair]])
            weights.append(1)

    edges = np.array(edges)
    G = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(count, count))

    labels = list(set(df.categories))
    categories = list(df.categories)
    for l in labels:
        print("Pagerank running for: ", l)
        personalized = np.zeros((count,))
        for i, cat in enumerate(categories):
            if cat == l:
                personalized[i] = 1
        pr = pagerank(G, p=0.85, personalize=personalized)
        temp_list = list(pr)[start:]
        sorted_temp_list = sorted(temp_list, reverse=True)
        args = np.argsort(temp_list)[::-1]
        top_auths = []
        for i in args:
            top_auths.append(num_auth_map[start + i])
        pickle.dump(top_auths, open(data_path + "top_auths/" + l + "_top_auths.pkl", "wb"))
