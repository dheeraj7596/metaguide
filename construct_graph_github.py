import pickle
from scipy import sparse
from parse_autophrase_output import decrypt
from nltk import word_tokenize
import numpy as np


def make_tag_map(df):
    count = len(df)
    tag_id = {}
    id_tag = {}

    tags_set = set()
    for auts in df.tags:
        tags_set.update(set(auts))

    for i, aut in enumerate(tags_set):
        tag_id[aut] = count
        id_tag[count] = aut
        count += 1
    return tag_id, id_tag, count


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


def make_user_map(df):
    count = len(df)
    users_set = set(df.user)
    user_id = {}
    id_user = {}

    for i, pub in enumerate(users_set):
        user_id[pub] = count
        id_user[count] = pub
        count += 1

    return user_id, id_user, count


def make_user_tag_map(df):
    count = len(df)
    user_tag_id = {}
    id_user_tag = {}

    total_keys = set()
    for i, row in df.iterrows():
        tags_list = row["tags"]
        user = row["user"]
        for tag in tags_list:
            total_keys.add((user, tag))

    for i, key in enumerate(total_keys):
        user_tag_id[key] = count
        id_user_tag[count] = key
        count += 1

    return user_tag_id, id_user_tag, count


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "github/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_bio_phrase_removed_stopwords.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_bio_map.pkl", "rb"))
    id_phrase_map = pickle.load(open(data_path + "id_phrase_bio_map.pkl", "rb"))

    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    existing_fnusts = set()
    for id in id_phrase_map:
        existing_fnusts.add("fnust" + str(id))

    fnust_id, id_fnust, fnust_graph_node_count = make_phrases_map(df, tokenizer, index_word, id_phrase_map)
    print(len(existing_fnusts - set(fnust_id.keys())))

    user_id, id_user, user_graph_node_count = make_user_map(df)

    tag_id, id_tag, tag_graph_node_count = make_tag_map(df)

    user_tag_id, id_user_tag, user_tag_graph_node_count = make_user_tag_map(df)

    edges = []
    weights = []
    for i, row in df.iterrows():
        abstract_str = row["text"]
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
        tags = row["tags"]
        for auth in tags:
            edges.append([i, tag_id[auth]])
            weights.append(1)
    edges = np.array(edges)
    G_tag = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                              shape=(tag_graph_node_count, tag_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        user = row["user"]
        edges.append([i, user_id[user]])
        weights.append(1)
    edges = np.array(edges)
    G_user = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                               shape=(user_graph_node_count, user_graph_node_count))

    edges = []
    weights = []
    for i, row in df.iterrows():
        tags = row["tags"]
        user = row["user"]
        for tag in tags:
            edges.append([i, user_tag_id[(user, tag)]])
            weights.append(1)
    edges = np.array(edges)
    G_user_tag = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                                   shape=(user_tag_graph_node_count, user_tag_graph_node_count))

    sparse.save_npz(data_path + "G_user_tag.npz", G_user_tag)
    sparse.save_npz(data_path + "G_phrase.npz", G_phrase)
    sparse.save_npz(data_path + "G_user.npz", G_user)
    sparse.save_npz(data_path + "G_tag.npz", G_tag)

    pickle.dump(fnust_id, open(data_path + "fnust_id.pkl", "wb"))
    pickle.dump(id_fnust, open(data_path + "id_fnust.pkl", "wb"))

    pickle.dump(user_id, open(data_path + "user_id.pkl", "wb"))
    pickle.dump(id_user, open(data_path + "id_user.pkl", "wb"))

    pickle.dump(tag_id, open(data_path + "tag_id.pkl", "wb"))
    pickle.dump(id_tag, open(data_path + "id_tag.pkl", "wb"))

    pickle.dump(user_tag_id, open(data_path + "user_tag_id.pkl", "wb"))
    pickle.dump(id_user_tag, open(data_path + "id_user_tag.pkl", "wb"))
