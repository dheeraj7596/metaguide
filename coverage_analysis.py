import pickle
from utils import *
import matplotlib.pyplot as plt


def get_idx_pairs(graph_dict, label_auth_dict):
    label_count_dict = {}
    auth_match_count = 0

    for l in graph_dict:
        graph = graph_dict[l]
        top_auth_pairs = [label_auth_dict[l][999]]
        count = 0
        for index in graph:
            existing_auth_pairs = graph[index]
            auth_pairs = list(set(top_auth_pairs).intersection(set(existing_auth_pairs)))
            auth_match_count += len(auth_pairs)
            if len(auth_pairs) > 0:
                count += 1
        label_count_dict[l] = count
    return label_count_dict, auth_match_count


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "arxiv_cs/"

    data_path = base_path + dataset
    auth_data_path = data_path + "top_auths/"
    df = pickle.load(open(data_path + "df_cs_2014_filtered.pkl", "rb"))
    graph_dict = pickle.load(open(data_path + "graph_dict.pkl", "rb"))
    labels = list(graph_dict.keys())
    ans = []
    for k in range(100, 1001, 100):
        k = 1000
        label_auth_dict = create_label_auth_dict(auth_data_path, labels, top_k=k)
        label_count_dict, auth_match_count = get_idx_pairs(graph_dict, label_auth_dict)
        count = 0
        for l in label_count_dict:
            count += label_count_dict[l]
            print(l, str(label_count_dict[l]) + "/" + str(len(graph_dict[l])))

        print("Total: ", str(count) + "/" + str(len(df)))
        print("Total author pair matches: ", auth_match_count)
        ans.append(count)
    plt.figure()
    plt.plot(range(100, 1001, 100), ans)
    plt.ylabel('# docs with atleast 1 author pair match')
    plt.xlabel('top-k')
    plt.savefig('./pair_match.png')
    pass
