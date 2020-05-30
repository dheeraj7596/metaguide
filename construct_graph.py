import pickle
from scipy import sparse
from parse_autophrase_output import decrypt
from nltk import word_tokenize
import numpy as np


def make_authors_map(df):
    count = len(df)
    author_id = {}
    id_author = {}

    authors_set = set()
    for auts in df.authors:
        authors = set(auts.split(","))
        for author in authors:
            if len(author) > 0:
                authors_set.add(author)

    for i, aut in enumerate(authors_set):
        author_id[aut] = count
        id_author[count] = aut
        count += 1
    return author_id, id_author, count


def make_years_map(df):
    count = len(df)
    year_id = {}
    id_year = {}

    years_set = set(df.year)

    for i, year in enumerate(years_set):
        year_id[year] = count
        id_year[count] = year
        count += 1
    return year_id, id_year, count


def detect_phrase(sentence, tokenizer, index_word, id_phrase_map, idx):
    tokens = tokenizer.texts_to_sequences([sentence])
    temp = []
    for tok in tokens[0]:
        try:
            id = decrypt(index_word[tok])
            if id == None or id not in id_phrase_map:
                if index_word[tok].startswith("fnust"):
                    num_str = index_word[tok][5:]
                    flag = 0
                    for index, char in enumerate(num_str):
                        if index >= 5:
                            break
                        try:
                            temp_int = int(char)
                            flag = 1
                        except:
                            break
                    if flag == 1:
                        if int(num_str[:index]) in id_phrase_map:
                            temp.append(index_word[tok])
                    else:
                        print(idx, index_word[tok])
            else:
                temp.append(index_word[tok])
        except Exception as e:
            pass
    return temp


def make_phrases_map(df, tokenizer, index_word, id_phrase_map):
    count = len(df)
    abstracts = list(df.abstract)
    fnust_id = {}
    id_fnust = {}

    for i, abstract in enumerate(abstracts):
        phrases = detect_phrase(abstract, tokenizer, index_word, id_phrase_map, i)
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


def create_year_graph(df, year_id, year_graph_node_count, data_path):
    edges = []
    weights = []
    for i, row in df.iterrows():
        year = row["year"]
        edges.append([i, year_id[year]])
        weights.append(1)
    edges = np.array(edges)
    G_year = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(year_graph_node_count, year_graph_node_count))
    sparse.save_npz(data_path + "G_year.npz", G_year)


def create_conf_graph(df, venue_id, venue_graph_node_count, data_path):
    edges = []
    weights = []
    for i, row in df.iterrows():
        conf = row["conf"]
        edges.append([i, venue_id[conf]])
        weights.append(1)
    edges = np.array(edges)
    G_conf = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(venue_graph_node_count, venue_graph_node_count))
    sparse.save_npz(data_path + "G_conf.npz", G_conf)


def create_author_graph(df, author_id, auth_graph_node_count, data_path):
    edges = []
    weights = []
    for i, row in df.iterrows():
        auth_str = row["authors"]
        authors = auth_str.split(",")
        for auth in authors:
            if len(auth) == 0:
                continue
            edges.append([i, author_id[auth]])
            weights.append(1)
    edges = np.array(edges)
    G_auth = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(auth_graph_node_count, auth_graph_node_count))
    sparse.save_npz(data_path + "G_auth_5.npz", G_auth)


def create_phrase_graph(df, tokenizer, index_word, id_phrase_map, fnust_id, fnust_graph_node_count, data_path):
    edges = []
    weights = []
    for i, row in df.iterrows():
        abstract_str = row["abstract"]
        phrases = detect_phrase(abstract_str, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            edges.append([i, fnust_id[ph]])
            weights.append(1)
    edges = np.array(edges)
    G_phrase = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                                 shape=(fnust_graph_node_count, fnust_graph_node_count))
    sparse.save_npz(data_path + "G_phrase.npz", G_phrase)


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "dblp/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_mapped_labels_phrase_removed_stopwords_test_thresh_5.pkl", "rb"))
    # tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    # phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))
    # id_phrase_map = {}
    # for ph in phrase_id_map:
    #     id_phrase_map[phrase_id_map[ph]] = ph
    # index_word = {}
    # for w in tokenizer.word_index:
    #     index_word[tokenizer.word_index[w]] = w
    #
    # existing_fnusts = set()
    # for id in id_phrase_map:
    #     existing_fnusts.add("fnust" + str(id))
    #
    # fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df, tokenizer, index_word, id_phrase_map)
    # print(len(existing_fnusts - set(fnust_id.keys())))
    author_id, id_author, auth_graph_node_count = make_authors_map(df)
    # venue_id, id_venue, venue_graph_node_count = make_venues_map(df)
    # year_id, id_year, year_graph_node_count = make_years_map(df)

    # create_phrase_graph(df, tokenizer, index_word, id_phrase_map, fnust_id, fnust_graph_node_count, data_path)

    create_author_graph(df, author_id, auth_graph_node_count, data_path)

    # create_conf_graph(df, venue_id, venue_graph_node_count, data_path)

    # create_year_graph(df, year_id, year_graph_node_count, data_path)
    #
    # pickle.dump(fnust_id, open(data_path + "fnust_id.pkl", "wb"))
    # pickle.dump(id_fnust, open(data_path + "id_fnust.pkl", "wb"))
    #
    # pickle.dump(venue_id, open(data_path + "venue_id.pkl", "wb"))
    # pickle.dump(id_venue, open(data_path + "id_venue.pkl", "wb"))

    pickle.dump(author_id, open(data_path + "author_id_5.pkl", "wb"))
    pickle.dump(id_author, open(data_path + "id_author_5.pkl", "wb"))

    # pickle.dump(year_id, open(data_path + "year_id.pkl", "wb"))
    # pickle.dump(id_year, open(data_path + "id_year.pkl", "wb"))
