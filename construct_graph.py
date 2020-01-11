import pickle
import itertools

if __name__ == "__main__":
    data_path = "./data/"
    df = pickle.load(open(data_path + "df_cs_2014_filtered.pkl", "rb"))
    label_id = pickle.load(open(data_path + "label_id.pkl", "rb"))
    id_label = pickle.load(open(data_path + "id_label.pkl", "rb"))
    author_id = pickle.load(open(data_path + "author_id.pkl", "rb"))
    id_author = pickle.load(open(data_path + "id_author.pkl", "rb"))

    graph_dict = {}
    for l in label_id:
        temp_df = df[df.categories.isin([l])].reset_index(drop=True)
        graph = {}
        for i, row in temp_df.iterrows():
            authors = row["authors"]
            ids = [author_id[aut] for aut in authors]
            assert len(ids) > 0
            if len(ids) >= 2:
                graph[i] = list(itertools.combinations(ids, 2))
            else:
                graph[i] = (ids[0],)

        graph_dict[l] = graph
    pickle.dump(graph_dict, open(data_path + "graph_dict.pkl", "wb"))
