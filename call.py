import pickle
import collections
from page_rank import powerIteration

if __name__ == "__main__":
    data_path = "/data4/dheeraj/metaguide/"
    df = pickle.load(open(data_path + "df_cs_2014_filtered.pkl", "rb"))
    graph_dict = pickle.load(open(data_path + "graph_dict.pkl", "rb"))
    label_top_10_auth_dict = {}

    for l in graph_dict:
        print("Pagerank for label: ", l)
        graph = graph_dict[l]
        label_top_10_auth_dict[l] = []
        edgeWeights = collections.defaultdict(lambda: collections.Counter())
        for doc_id in graph:
            for pair in graph[doc_id]:
                edgeWeights[doc_id][pair] += 1.0
                edgeWeights[pair][doc_id] += 1.0

        wordProbabilities = powerIteration(edgeWeights, rsp=0.15)
        sorted_wordprobs = wordProbabilities.sort_values(ascending=False)
        indices = list(sorted_wordprobs.index)
        count = 0
        for ind in indices:
            if count == 10:
                break
            if type(ind) == tuple:
                label_top_10_auth_dict[l].append(ind)
                count += 1

    pickle.dump(label_top_10_auth_dict, open(data_path + "label_top_10_auth_dict.pkl"))
