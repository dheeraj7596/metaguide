import pickle
import numpy as np
from scipy import sparse
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power


def make_auth_pair_map(graph):
    start = len(graph)
    auth_num_map = {}
    num_auth_map = {}
    auth_pairs_set = set()
    for doc_id in graph:
        auth_pairs_set.update(set(graph[doc_id]))

    for pair in auth_pairs_set:
        auth_num_map[pair] = start
        num_auth_map[start] = pair
        start += 1
    return auth_num_map, num_auth_map, start


if __name__ == "__main__":
    data_path = "/data4/dheeraj/metaguide/"
    df = pickle.load(open(data_path + "df_cs_2014_filtered.pkl", "rb"))
    graph_dict = pickle.load(open(data_path + "graph_dict.pkl", "rb"))

    for l in graph_dict:
        print("Pagerank for label: ", l)
        graph = graph_dict[l]
        start = len(graph)
        auth_num_map, num_auth_map, count = make_auth_pair_map(graph)

        edges = []
        weights = []
        for doc_id in graph:
            for pair in graph[doc_id]:
                edges.append([doc_id, auth_num_map[pair]])
                weights.append(1)

        edges = np.array(edges)
        G = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(count, count))
        pr = pagerank(G, p=0.85)
        temp_list = list(pr)[start:]
        args = np.argsort(temp_list)[::-1]
        top_auths = []
        for i in args:
            top_auths.append(num_auth_map[start + i])
        pickle.dump(top_auths, open(data_path + "top_auths/" + l + "_top_auths.pkl", "wb"))
