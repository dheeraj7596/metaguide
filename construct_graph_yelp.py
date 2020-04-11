import pickle
from scipy import sparse
from parse_autophrase_output import decrypt
from nltk import word_tokenize
from data.yelp.attribute_utils import *
import numpy as np


def make_authors_map(df):
    count = len(df)
    author_id = {}
    id_author = {}

    authors_set = set()
    for auts in df.Users:
        authors_set.update(set(auts))

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
    abstracts = list(df.Review)
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


def make_attributes_map(df):
    count = len(df)
    attr_id = {}
    id_attr = {}

    total_keys = set()
    for i, row in df.iterrows():
        row_keys = get_all_keys(row)
        total_keys.update(row_keys)

    for i, key in enumerate(total_keys):
        attr_id[key] = count
        id_attr[count] = key
        count += 1

    return attr_id, id_attr, count


def make_author_attributes_map(df):
    count = len(df)
    author_attr_id = {}
    id_author_attr = {}

    total_keys = set()
    for i, row in df.iterrows():
        auth_list = row["Users"]
        row_keys = get_all_keys(row)
        for auth in auth_list:
            for key in row_keys:
                total_keys.add((auth, key))

    for i, key in enumerate(total_keys):
        author_attr_id[key] = count
        id_author_attr[count] = key
        count += 1

    return author_attr_id, id_author_attr, count


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "yelp/"

    data_path = base_path + dataset
    df = pickle.load(
        open(data_path + "business_1review_shortlisted_thresh_3_phrase_removed_stopwords.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_map.pkl", "rb"))

    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    existing_fnusts = set()
    for id in id_phrase_map:
        existing_fnusts.add("fnust" + str(id))

    fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df, tokenizer, index_word, id_phrase_map)
    print(len(existing_fnusts - set(fnust_id.keys())))

    author_id, id_author, auth_graph_node_count = make_authors_map(df)

    attr_id, id_attr, attr_graph_node_count = make_attributes_map(df)

    author_attr_id, id_author_attr, author_attr_graph_node_count = make_author_attributes_map(df)

    edges = []
    weights = []
    for i, row in df.iterrows():
        abstract_str = row["Review"]
        phrases = detect_phrase(abstract_str, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            edges.append([i, fnust_id[ph]])
            weights.append(1)
    edges = np.array(edges)
    G_phrase = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                                 shape=(fnust_graph_node_count, fnust_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        authors = row["Users"]
        for auth in authors:
            edges.append([i, author_id[auth]])
            weights.append(1)
    edges = np.array(edges)
    G_auth = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(auth_graph_node_count, auth_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        row_keys = get_all_keys(row)
        for key in row_keys:
            edges.append([i, attr_id[key]])
            weights.append(1)
    edges = np.array(edges)
    G_attr = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(attr_graph_node_count, attr_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        row_keys = get_all_keys(row)
        auth_list = row["Users"]
        for auth in auth_list:
            for key in row_keys:
                edges.append([i, author_attr_id[(auth, key)]])
                weights.append(1)
    edges = np.array(edges)
    G_auth_attr = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                                    shape=(author_attr_graph_node_count, author_attr_graph_node_count))

    sparse.save_npz(data_path + "G_phrase_1review_shortlisted_thresh_3.npz", G_phrase)
    sparse.save_npz(data_path + "G_auth_1review_shortlisted_thresh_3.npz", G_auth)
    sparse.save_npz(data_path + "G_attr_1review_shortlisted_thresh_3.npz", G_attr)
    sparse.save_npz(data_path + "G_auth_attr_1review_shortlisted_thresh_3.npz", G_auth_attr)

    pickle.dump(fnust_id, open(data_path + "fnust_id_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(id_fnust, open(data_path + "id_fnust_1review_shortlisted_thresh_3.pkl", "wb"))

    pickle.dump(author_id, open(data_path + "author_id_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(id_author, open(data_path + "id_author_1review_shortlisted_thresh_3.pkl", "wb"))

    pickle.dump(attr_id, open(data_path + "attr_id_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(id_attr, open(data_path + "id_attr_1review_shortlisted_thresh_3.pkl", "wb"))

    pickle.dump(author_attr_id, open(data_path + "author_attr_id_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(id_author_attr, open(data_path + "id_author_attr_1review_shortlisted_thresh_3.pkl", "wb"))
