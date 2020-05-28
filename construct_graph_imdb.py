import pickle
from scipy import sparse
from parse_autophrase_output import decrypt
from nltk import word_tokenize
import numpy as np
import itertools


def make_unigram_map(df, col):
    count = len(df)
    col_id = {}
    id_col = {}

    cols_set = set()
    for entities in df[col]:
        cols_set.update(entities)

    for i, entity in enumerate(cols_set):
        col_id[entity] = count
        id_col[count] = entity
        count += 1
    return col_id, id_col, count


def make_bigram_map(df, col1, col2):
    count = len(df)
    col_id = {}
    id_col = {}

    cols_set = set()
    for i, row in df.iterrows():
        col1_entities = row[col1]
        col2_entities = row[col2]
        cols_set.update(set(itertools.product(col1_entities, col2_entities)))

    for i, entity in enumerate(cols_set):
        col_id[entity] = count
        id_col[count] = entity
        count += 1
    return col_id, id_col, count


def make_isadult_map(df):
    count = len(df)
    isadult_id = {}
    id_isadult = {}

    isadult_set = set(df["isAdult"])

    for i, ent in enumerate(isadult_set):
        isadult_id[ent] = count
        id_isadult[count] = ent
        count += 1
    return isadult_id, id_isadult, count


def make_dir_adult_map(df):
    count = len(df)
    col_id = {}
    id_col = {}

    cols_set = set()
    for i, row in df.iterrows():
        col1_entities = row["director"]
        col2_entities = {row["isAdult"]}
        cols_set.update(set(itertools.product(col1_entities, col2_entities)))

    for i, entity in enumerate(cols_set):
        col_id[entity] = count
        id_col[count] = entity
        count += 1
    return col_id, id_col, count


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


def create_phrase_graph(df, fnust_id, tokenizer, index_word, id_phrase_map, fnust_graph_node_count):
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
    return G_phrase


def create_unigram_graph(df, col, col_id, graph_node_count):
    edges = []
    weights = []
    for i, row in df.iterrows():
        col_entities = row[col]
        for entity in col_entities:
            edges.append([i, col_id[entity]])
            weights.append(1)
    edges = np.array(edges)
    G_col = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                              shape=(graph_node_count, graph_node_count))
    return G_col


def create_bigram_graph(df, col1, col2, col_id, graph_node_count):
    edges = []
    weights = []
    for i, row in df.iterrows():
        col1_entities = row[col1]
        col2_entities = row[col2]
        cols_set = set(itertools.product(col1_entities, col2_entities))
        for entity in cols_set:
            edges.append([i, col_id[entity]])
            weights.append(1)
    edges = np.array(edges)
    G_col = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                              shape=(graph_node_count, graph_node_count))
    return G_col


def create_dir_adult_graph(df, dir_adult_id, dir_adult_graph_node_count):
    edges = []
    weights = []
    for i, row in df.iterrows():
        col1_entities = row["director"]
        col2_entities = {row["isAdult"]}
        cols_set = set(itertools.product(col1_entities, col2_entities))
        for entity in cols_set:
            edges.append([i, dir_adult_id[entity]])
            weights.append(1)
    edges = np.array(edges)
    G_col = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                              shape=(dir_adult_graph_node_count, dir_adult_graph_node_count))
    return G_col


def create_adult_graph(df, adult_id, adult_graph_node_count):
    edges = []
    weights = []
    for i, row in df.iterrows():
        entity = row["isAdult"]
        edges.append([i, adult_id[entity]])
        weights.append(1)
    edges = np.array(edges)
    G_col = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])),
                              shape=(adult_graph_node_count, adult_graph_node_count))
    return G_col


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "imdb/"

    data_path = base_path + dataset
    df = pickle.load(open(data_path + "df_summary_top6_title_summary_all_reviews_removed_stopwords_metadata.pkl", "rb"))

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

    # unigram metadata
    adult_id, id_adult, adult_graph_node_count = make_isadult_map(df)

    actor_id, id_actor, actor_graph_node_count = make_unigram_map(df, "actor")

    actress_id, id_actress, actress_graph_node_count = make_unigram_map(df, "actress")

    producer_id, id_producer, producer_graph_node_count = make_unigram_map(df, "producer")

    writer_id, id_writer, writer_graph_node_count = make_unigram_map(df, "writer")

    director_id, id_director, director_graph_node_count = make_unigram_map(df, "director")

    composer_id, id_composer, composer_graph_node_count = make_unigram_map(df, "composer")

    cinematographer_id, id_cinematographer, cinematographer_graph_node_count = make_unigram_map(df, "cinematographer")

    editor_id, id_editor, editor_graph_node_count = make_unigram_map(df, "editor")

    prod_designer_id, id_prod_designer, prod_designer_graph_node_count = make_unigram_map(df, "prod_designer")

    # bigram metadata
    dir_adult_id, id_dir_adult, dir_adult_graph_node_count = make_dir_adult_map(df)

    dir_actor_id, id_dir_actor, dir_actor_graph_node_count = make_bigram_map(df, "director", "actor")

    dir_actress_id, id_dir_actress, dir_actress_graph_node_count = make_bigram_map(df, "director", "actress")

    dir_producer_id, id_dir_producer, dir_producer_graph_node_count = make_bigram_map(df, "director", "producer")

    dir_writer_id, id_dir_writer, dir_writer_graph_node_count = make_bigram_map(df, "director", "writer")

    dir_composer_id, id_dir_composer, dir_composer_graph_node_count = make_bigram_map(df, "director", "composer")

    dir_cinematographer_id, id_dir_cinematographer, dir_cinematographer_graph_node_count = make_bigram_map(df,
                                                                                                           "director",
                                                                                                           "cinematographer")

    dir_editor_id, id_dir_editor, dir_editor_graph_node_count = make_bigram_map(df, "director", "editor")

    dir_prod_designer_id, id_dir_prod_designer, dir_prod_designer_graph_node_count = make_bigram_map(df, "director",
                                                                                                     "prod_designer")
    actor_actress_id, id_actor_actress, actor_actress_graph_node_count = make_bigram_map(df, "actor", "actress")

    # unigram graphs
    G_phrase = create_phrase_graph(df, fnust_id, tokenizer, index_word, id_phrase_map, fnust_graph_node_count)
    G_adult = create_adult_graph(df, adult_id, adult_graph_node_count)
    G_actor = create_unigram_graph(df, "actor", actor_id, actor_graph_node_count)
    G_actress = create_unigram_graph(df, "actress", actress_id, actress_graph_node_count)
    G_producer = create_unigram_graph(df, "producer", producer_id, producer_graph_node_count)
    G_writer = create_unigram_graph(df, "writer", writer_id, writer_graph_node_count)
    G_director = create_unigram_graph(df, "director", director_id, director_graph_node_count)
    G_composer = create_unigram_graph(df, "composer", composer_id, composer_graph_node_count)
    G_cinematographer = create_unigram_graph(df, "cinematographer", cinematographer_id,
                                             cinematographer_graph_node_count)
    G_editor = create_unigram_graph(df, "editor", editor_id, editor_graph_node_count)
    G_prod_designer = create_unigram_graph(df, "prod_designer", prod_designer_id, prod_designer_graph_node_count)

    # bigram graphs
    G_dir_adult = create_dir_adult_graph(df, dir_adult_id, dir_adult_graph_node_count)
    G_dir_actor = create_bigram_graph(df, "director", "actor", dir_actor_id, dir_actor_graph_node_count)
    G_dir_actress = create_bigram_graph(df, "director", "actress", dir_actress_id, dir_actress_graph_node_count)
    G_dir_producer = create_bigram_graph(df, "director", "producer", dir_producer_id, dir_producer_graph_node_count)
    G_dir_writer = create_bigram_graph(df, "director", "writer", dir_writer_id, dir_writer_graph_node_count)
    G_dir_composer = create_bigram_graph(df, "director", "composer", dir_composer_id, dir_composer_graph_node_count)
    G_dir_cinematographer = create_bigram_graph(df, "director", "cinematographer", dir_cinematographer_id,
                                                dir_cinematographer_graph_node_count)
    G_dir_editor = create_bigram_graph(df, "director", "editor", dir_editor_id, dir_editor_graph_node_count)
    G_dir_prod_designer = create_bigram_graph(df, "director", "prod_designer", dir_prod_designer_id,
                                              dir_prod_designer_graph_node_count)
    G_actor_actress = create_bigram_graph(df, "actor", "actress", actor_actress_id, actor_actress_graph_node_count)

    # pickle dumps
    dump_path = data_path + "graph/"
    sparse.save_npz(dump_path + "G_phrase.npz", G_phrase)
    sparse.save_npz(dump_path + "G_adult.npz", G_adult)
    sparse.save_npz(dump_path + "G_actor.npz", G_actor)
    sparse.save_npz(dump_path + "G_actress.npz", G_actress)
    sparse.save_npz(dump_path + "G_producer.npz", G_producer)
    sparse.save_npz(dump_path + "G_writer.npz", G_writer)
    sparse.save_npz(dump_path + "G_director.npz", G_director)
    sparse.save_npz(dump_path + "G_composer.npz", G_composer)
    sparse.save_npz(dump_path + "G_cinematographer.npz", G_cinematographer)
    sparse.save_npz(dump_path + "G_editor.npz", G_editor)
    sparse.save_npz(dump_path + "G_prod_designer.npz", G_prod_designer)

    sparse.save_npz(dump_path + "G_dir_adult.npz", G_dir_adult)
    sparse.save_npz(dump_path + "G_dir_actor.npz", G_dir_actor)
    sparse.save_npz(dump_path + "G_dir_actress.npz", G_dir_actress)
    sparse.save_npz(dump_path + "G_dir_producer.npz", G_dir_producer)
    sparse.save_npz(dump_path + "G_dir_writer.npz", G_dir_writer)
    sparse.save_npz(dump_path + "G_dir_composer.npz", G_dir_composer)
    sparse.save_npz(dump_path + "G_dir_cinematographer.npz", G_dir_cinematographer)
    sparse.save_npz(dump_path + "G_dir_editor.npz", G_dir_editor)
    sparse.save_npz(dump_path + "G_dir_prod_designer.npz", G_dir_prod_designer)
    sparse.save_npz(dump_path + "G_actor_actress.npz", G_actor_actress)

    pickle.dump(fnust_id, open(dump_path + "fnust_id.pkl", "wb"))
    pickle.dump(id_fnust, open(dump_path + "id_fnust.pkl", "wb"))

    pickle.dump(adult_id, open(dump_path + "adult_id.pkl", "wb"))
    pickle.dump(id_adult, open(dump_path + "id_adult.pkl", "wb"))

    pickle.dump(actor_id, open(dump_path + "actor_id.pkl", "wb"))
    pickle.dump(id_actor, open(dump_path + "id_actor.pkl", "wb"))

    pickle.dump(actress_id, open(dump_path + "actress_id.pkl", "wb"))
    pickle.dump(id_actress, open(dump_path + "id_actress.pkl", "wb"))

    pickle.dump(producer_id, open(dump_path + "producer_id.pkl", "wb"))
    pickle.dump(id_producer, open(dump_path + "id_producer.pkl", "wb"))

    pickle.dump(writer_id, open(dump_path + "writer_id.pkl", "wb"))
    pickle.dump(id_writer, open(dump_path + "id_writer.pkl", "wb"))

    pickle.dump(director_id, open(dump_path + "director_id.pkl", "wb"))
    pickle.dump(id_director, open(dump_path + "id_director.pkl", "wb"))

    pickle.dump(composer_id, open(dump_path + "composer_id.pkl", "wb"))
    pickle.dump(id_composer, open(dump_path + "id_composer.pkl", "wb"))

    pickle.dump(cinematographer_id, open(dump_path + "cinematographer_id.pkl", "wb"))
    pickle.dump(id_cinematographer, open(dump_path + "id_cinematographer.pkl", "wb"))

    pickle.dump(editor_id, open(dump_path + "editor_id.pkl", "wb"))
    pickle.dump(id_editor, open(dump_path + "id_editor.pkl", "wb"))

    pickle.dump(prod_designer_id, open(dump_path + "prod_designer_id.pkl", "wb"))
    pickle.dump(id_prod_designer, open(dump_path + "id_prod_designer.pkl", "wb"))

    pickle.dump(dir_adult_id, open(dump_path + "dir_adult_id.pkl", "wb"))
    pickle.dump(id_dir_adult, open(dump_path + "id_dir_adult.pkl", "wb"))

    pickle.dump(dir_actor_id, open(dump_path + "dir_actor_id.pkl", "wb"))
    pickle.dump(id_dir_actor, open(dump_path + "id_dir_actor.pkl", "wb"))

    pickle.dump(dir_actress_id, open(dump_path + "dir_actress_id.pkl", "wb"))
    pickle.dump(id_dir_actress, open(dump_path + "id_dir_actress.pkl", "wb"))

    pickle.dump(dir_producer_id, open(dump_path + "dir_producer_id.pkl", "wb"))
    pickle.dump(id_dir_producer, open(dump_path + "id_dir_producer.pkl", "wb"))

    pickle.dump(dir_writer_id, open(dump_path + "dir_writer_id.pkl", "wb"))
    pickle.dump(id_dir_writer, open(dump_path + "id_dir_writer.pkl", "wb"))

    pickle.dump(dir_composer_id, open(dump_path + "dir_composer_id.pkl", "wb"))
    pickle.dump(id_dir_composer, open(dump_path + "id_dir_composer.pkl", "wb"))

    pickle.dump(dir_cinematographer_id, open(dump_path + "dir_cinematographer_id.pkl", "wb"))
    pickle.dump(id_dir_cinematographer, open(dump_path + "id_dir_cinematographer.pkl", "wb"))

    pickle.dump(dir_editor_id, open(dump_path + "dir_editor_id.pkl", "wb"))
    pickle.dump(id_dir_editor, open(dump_path + "id_dir_editor.pkl", "wb"))

    pickle.dump(dir_prod_designer_id, open(dump_path + "dir_prod_designer_id.pkl", "wb"))
    pickle.dump(id_dir_prod_designer, open(dump_path + "id_dir_prod_designer.pkl", "wb"))

    pickle.dump(actor_actress_id, open(dump_path + "actor_actress_id.pkl", "wb"))
    pickle.dump(id_actor_actress, open(dump_path + "id_actor_actress.pkl", "wb"))
