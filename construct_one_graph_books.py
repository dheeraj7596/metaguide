import pickle
from scipy import sparse
from parse_autophrase_output import decrypt
from nltk import word_tokenize
import numpy as np


def make_authors_map(df, count):
    author_id = {}
    id_author = {}

    authors_set = set()
    for auts in df.authors:
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


def make_phrases_map(df, tokenizer, index_word, id_phrase_map, count):
    abstracts = list(df.text)
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


def make_pub_map(df, count):
    pubs_set = set(df.publisher)
    pub_id = {}
    id_pub = {}

    for i, pub in enumerate(pubs_set):
        pub_id[pub] = count
        id_pub[count] = pub
        count += 1

    return pub_id, id_pub, count


def make_pub_year_map(df, count):
    pub_year_id = {}
    id_pub_year = {}

    total_keys = set()
    for i, row in df.iterrows():
        year = row["publication_year"]
        pub = row["publisher"]
        total_keys.add((pub, year))

    for i, key in enumerate(total_keys):
        pub_year_id[key] = count
        id_pub_year[count] = key
        count += 1

    return pub_year_id, id_pub_year, count


def make_author_pub_map(df, count):
    author_pub_id = {}
    id_author_pub = {}

    total_keys = set()
    for i, row in df.iterrows():
        auth_list = row["authors"]
        pub = row["publisher"]
        for auth in auth_list:
            total_keys.add((auth, pub))

    for i, key in enumerate(total_keys):
        author_pub_id[key] = count
        id_author_pub[count] = key
        count += 1

    return author_pub_id, id_author_pub, count


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "books/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_phrase_removed_stopwords.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_map.pkl", "rb"))

    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    existing_fnusts = set()
    for id in id_phrase_map:
        existing_fnusts.add("fnust" + str(id))

    fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df, tokenizer, index_word, id_phrase_map, len(df))
    print(len(existing_fnusts - set(fnust_id.keys())))

    author_id, id_author, auth_graph_node_count = make_authors_map(df, fnust_graph_node_count)

    pub_id, id_pub, pub_graph_node_count = make_pub_map(df, auth_graph_node_count)

    author_pub_id, id_author_pub, author_pub_graph_node_count = make_author_pub_map(df, pub_graph_node_count)

    pub_year_id, id_pub_year, pub_year_graph_node_count = make_pub_year_map(df, author_pub_graph_node_count)

    edges = []
    weights = []
    for i, row in df.iterrows():
        abstract_str = row["text"]
        phrases = detect_phrase(abstract_str, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            edges.append([i, fnust_id[ph]])
            weights.append(1)

        authors = row["authors"]
        for auth in authors:
            edges.append([i, author_id[auth]])
            weights.append(1)

        pub = row["publisher"]
        edges.append([i, pub_id[pub]])
        weights.append(1)

        auth_list = row["authors"]
        pub = row["publisher"]
        for auth in auth_list:
            edges.append([i, author_pub_id[(auth, pub)]])
            weights.append(1)

        year = row["publication_year"]
        pub = row["publisher"]
        edges.append([i, pub_year_id[(pub, year)]])
        weights.append(1)

    edges = np.array(edges)
    G_all = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                              shape=(pub_year_graph_node_count, pub_year_graph_node_count))

    sparse.save_npz(data_path + "G_all.npz", G_all)

    pickle.dump(fnust_id, open(data_path + "fnust_id_all.pkl", "wb"))
    pickle.dump(id_fnust, open(data_path + "id_fnust_all.pkl", "wb"))

    pickle.dump(author_id, open(data_path + "author_id_all.pkl", "wb"))
    pickle.dump(id_author, open(data_path + "id_author_all.pkl", "wb"))

    pickle.dump(pub_id, open(data_path + "pub_id_all.pkl", "wb"))
    pickle.dump(id_pub, open(data_path + "id_pub_all.pkl", "wb"))

    pickle.dump(author_pub_id, open(data_path + "author_pub_id_all.pkl", "wb"))
    pickle.dump(id_author_pub, open(data_path + "id_author_pub_all.pkl", "wb"))

    pickle.dump(pub_year_id, open(data_path + "pub_year_id_all.pkl", "wb"))
    pickle.dump(id_pub_year, open(data_path + "id_pub_year_all.pkl", "wb"))
