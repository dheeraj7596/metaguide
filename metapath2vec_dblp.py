from construct_graph import make_phrases_map, make_authors_map, detect_phrase, make_years_map
import numpy as np
from stellargraph.core.indexed_array import IndexedArray
from stellargraph.data import UniformRandomMetaPathWalk
from sklearn.metrics import classification_report
from stellargraph import StellarGraph
from gensim.models import Word2Vec
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from cocube_utils_beta import get_distinct_labels
from coc_data_utils import get_label_term_json, modify_phrases


def get_graph_metapaths(df, tokenizer, id_phrase_map):
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    existing_fnusts = set()
    for id in id_phrase_map:
        existing_fnusts.add("fnust" + str(id))

    doc_index = []
    for i in range(len(df)):
        doc_index.append("doc" + str(i))
    doc_nodes = IndexedArray(np.array([[-1]] * len(df)), index=doc_index)

    fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df, tokenizer, index_word, id_phrase_map)
    phrase_index = []
    for i in range(len(df), fnust_graph_node_count):
        phrase_index.append("phrase" + str(i))
    phrase_nodes = IndexedArray(np.array([[-1]] * len(phrase_index)), index=phrase_index)
    print(len(existing_fnusts - set(fnust_id.keys())))

    author_id, id_author, auth_graph_node_count = make_authors_map(df, fnust_graph_node_count)
    author_index = []
    for i in range(fnust_graph_node_count, auth_graph_node_count):
        author_index.append("author" + str(i))
    author_nodes = IndexedArray(np.array([[-1]] * len(author_index)), index=author_index)

    year_id, id_year, year_graph_node_count = make_years_map(df, auth_graph_node_count)
    year_index = []
    for i in range(auth_graph_node_count, year_graph_node_count):
        year_index.append("year" + str(i))
    year_nodes = IndexedArray(np.array([[-1]] * len(year_index)), index=year_index)

    source_nodes_list = []
    target_nodes_list = []
    for i, row in df.iterrows():
        abstract_str = row["abstract"]
        phrases = detect_phrase(abstract_str, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            source_nodes_list.append("doc" + str(i))
            target_nodes_list.append("phrase" + str(fnust_id[ph]))

        auth_str = row["authors"]
        authors = auth_str.split(",")
        for auth in authors:
            if len(auth) == 0:
                continue
            source_nodes_list.append("doc" + str(i))
            target_nodes_list.append("author" + str(author_id[auth]))

        year = row["year"]
        source_nodes_list.append("doc" + str(i))
        target_nodes_list.append("year" + str(year_id[year]))

    edges = pd.DataFrame({
        "source": source_nodes_list,
        "target": target_nodes_list
    })

    graph = StellarGraph({"doc": doc_nodes, "phrase": phrase_nodes, "author": author_nodes, "year": year_nodes}, edges)
    metapaths = [
        ["doc", "phrase", "doc"],
        ["doc", "author", "doc"],
        ["doc", "year", "doc"],
    ]
    return graph, metapaths


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
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

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["abstract"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(index)
            y_true.append(label)
    return X, y, y_true


if __name__ == "__main__":
    # base_path = "./data/"
    base_path = "/data4/dheeraj/metaguide/"
    dataset = "dblp/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_mapped_labels_phrase_removed_stopwords_test_thresh_3.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_map.pkl", "rb"))

    labels, label_to_index, index_to_label = get_distinct_labels(df)
    label_term_dict = get_label_term_json(data_path + "seedwords.json")
    label_term_dict = modify_phrases(label_term_dict, phrase_id_map)

    graph, metapaths = get_graph_metapaths(df, tokenizer, id_phrase_map)

    print(
        "Number of nodes {} and number of edges {} in graph.".format(
            graph.number_of_nodes(), graph.number_of_edges()
        )
    )

    rw = UniformRandomMetaPathWalk(graph)
    walks = rw.run(
        nodes=list(graph.nodes()),  # root nodes
        length=5,  # maximum length of a random walk
        n=5,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
    )
    print("Number of random walks: {}".format(len(walks)))

    model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=2, iter=10)
    print("Embeddings shape: ", model.wv.vectors.shape)

    node_ids = model.wv.index2word  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    model.save(data_path + "node_embeddings.model")

    node_targets = [graph.node_type(node_id) for node_id in node_ids]
    num_doc_embeds = 0
    for k in node_targets:
        if k == "doc":
            num_doc_embeds += 1
    assert num_doc_embeds == len(df)

    doc_embeddings = []
    for i in range(len(df)):
        doc_embeddings.append(node_embeddings[node_ids.index("doc" + str(i))])

    X_inds, y, y_true = generate_pseudo_labels(df, labels, label_term_dict, tokenizer)

    X = []
    for i in X_inds:
        X.append(doc_embeddings[i])

    clf = LogisticRegression(random_state=0, max_iter=100000).fit(X, y)
    preds = clf.predict(doc_embeddings)
    print(classification_report(df["label"], preds))
