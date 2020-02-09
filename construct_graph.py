import pickle
from scipy import sparse
from nltk import word_tokenize
import numpy as np


def make_authors_map(df):
    count = len(df)
    author_id = {}
    id_author = {}

    authors_set = set()
    for auts in df.authors:
        authors_set.update(set(auts.split(",")))

    for i, aut in enumerate(authors_set):
        author_id[aut] = count
        id_author[count] = aut
        count += 1
    return author_id, id_author, count


def detect_phrase(sentence):
    words = word_tokenize(sentence)
    temp = []
    for word in words:
        if word.startswith('fnust'):
            temp.append(word)
    return temp


def make_phrases_map(df):
    count = len(df)
    abstracts = list(df.abstract)
    fnust_id = {}
    id_fnust = {}

    for i, abstract in enumerate(abstracts):
        phrases = detect_phrase(abstract)
        for ph in phrases:
            try:
                temp = fnust_id[ph]
            except:
                fnust_id[ph] = count
                id_fnust[count] = ph
                count += 1

    return fnust_id, id_fnust, count


def make_venues_map(df):
    count = len(df)
    venues_set = set(df.conf)
    venue_id = {}
    id_venue = {}

    for i, venue in enumerate(venues_set):
        venue_id[venue] = count
        id_venue[count] = venue
        count += 1

    return venue_id, id_venue, count


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "dblp/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_mapped_labels_phrase_removed_stopwords.pkl", "rb"))

    author_id, id_author, auth_graph_node_count = make_authors_map(df)
    venue_id, id_venue, venue_graph_node_count = make_venues_map(df)
    fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df)

    edges = []
    weights = []
    for i, row in df.iterrows():
        abstract_str = row["abstract"]
        phrases = detect_phrase(abstract_str)
        for ph in phrases:
            edges.append([i, fnust_id[ph]])
            weights.append(1)
    edges = np.array(edges)
    G_phrase = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                                 shape=(fnust_graph_node_count, fnust_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        auth_str = row["authors"]
        authors = auth_str.split(",")
        for auth in authors:
            edges.append([i, author_id[auth]])
            weights.append(1)
    edges = np.array(edges)
    G_auth = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(auth_graph_node_count, auth_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        conf = row["conf"]
        edges.append([i, venue_id[conf]])
        weights.append(1)
    edges = np.array(edges)
    G_conf = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(venue_graph_node_count, venue_graph_node_count))

    sparse.save_npz(data_path + "G_phrase.npz", G_phrase)
    sparse.save_npz(data_path + "G_auth.npz", G_auth)
    sparse.save_npz(data_path + "G_conf.npz", G_conf)

    pickle.dump(fnust_id, open(data_path + "fnust_id.pkl", "wb"))
    pickle.dump(id_fnust, open(data_path + "id_fnust.pkl", "wb"))

    pickle.dump(venue_id, open(data_path + "venue_id.pkl", "wb"))
    pickle.dump(id_venue, open(data_path + "id_venue.pkl", "wb"))

    pickle.dump(author_id, open(data_path + "author_id.pkl", "wb"))
    pickle.dump(id_author, open(data_path + "id_author.pkl", "wb"))
