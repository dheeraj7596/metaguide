from construct_graph import detect_phrase
import pickle
import itertools


def create_phrase_doc_id_map(df, tokenizer, phrase_id_map):
    phrase_docid = {}

    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    id_phrase_map = {}
    for ph in phrase_id_map:
        id_phrase_map[phrase_id_map[ph]] = ph

    for i, row in df.iterrows():
        abstract_str = row["text"]
        phrases = detect_phrase(abstract_str, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            try:
                phrase_docid[ph].add(i)
            except:
                phrase_docid[ph] = {i}
    return phrase_docid


def create_unigram_doc_id_map(df, col):
    ent_docid = {}
    for i, row in df.iterrows():
        auts = row[col]
        for ent in auts:
            try:
                ent_docid[ent].add(i)
            except:
                ent_docid[ent] = {i}
    return ent_docid


def create_isadult_doc_id_map(df):
    adult_docid_map = {}
    for i, row in df.iterrows():
        pub = row["isAdult"]
        try:
            adult_docid_map[pub].add(i)
        except:
            adult_docid_map[pub] = {i}
    return adult_docid_map


def create_dir_isadult_doc_id_map(df):
    dir_isadult_docid_map = {}
    for i, row in df.iterrows():
        auts = row["director"]
        pub = {row["isAdult"]}
        cols_set = set(itertools.product(auts, pub))
        for aut in cols_set:
            try:
                dir_isadult_docid_map[aut].add(i)
            except:
                dir_isadult_docid_map[aut] = {i}
    return dir_isadult_docid_map


def create_bigram_doc_id_map(df, col1, col2):
    bigram_ent_docid_map = {}
    for i, row in df.iterrows():
        col1_entities = row[col1]
        col2_entities = row[col2]
        cols_set = set(itertools.product(col1_entities, col2_entities))
        for aut in cols_set:
            try:
                bigram_ent_docid_map[aut].add(i)
            except:
                bigram_ent_docid_map[aut] = {i}
    return bigram_ent_docid_map


if __name__ == "__main__":
    data_path = "./"

    df = pickle.load(open(data_path + "df_summary_top6_title_summary_all_reviews_removed_stopwords_metadata.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))

    phrase_docid = create_phrase_doc_id_map(df, tokenizer, phrase_id_map)

    adult_docid = create_isadult_doc_id_map(df)
    actor_docid = create_unigram_doc_id_map(df, "actor")
    actress_docid = create_unigram_doc_id_map(df, "actress")
    producer_docid = create_unigram_doc_id_map(df, "producer")
    writer_docid = create_unigram_doc_id_map(df, "writer")
    director_docid = create_unigram_doc_id_map(df, "director")
    composer_docid = create_unigram_doc_id_map(df, "composer")
    cinematographer_docid = create_unigram_doc_id_map(df, "cinematographer")
    editor_docid = create_unigram_doc_id_map(df, "editor")
    prod_designer_docid = create_unigram_doc_id_map(df, "prod_designer")

    dir_adult_docid = create_dir_isadult_doc_id_map(df)
    dir_actor_docid = create_bigram_doc_id_map(df, "director", "actor")
    dir_actress_docid = create_bigram_doc_id_map(df, "director", "actress")
    dir_producer_docid = create_bigram_doc_id_map(df, "director", "producer")
    dir_writer_docid = create_bigram_doc_id_map(df, "director", "writer")
    dir_composer_docid = create_bigram_doc_id_map(df, "director", "composer")
    dir_cinematographer_docid = create_bigram_doc_id_map(df, "director", "cinematographer")
    dir_editor_docid = create_bigram_doc_id_map(df, "director", "editor")
    dir_prod_designer_docid = create_bigram_doc_id_map(df, "director", "prod_designer")
    actor_actress_docid = create_bigram_doc_id_map(df, "actor", "actress")
    
    dump_path = data_path + "graph/"
    pickle.dump(phrase_docid, open(dump_path + "phrase_docid_map.pkl", "wb"))
    pickle.dump(adult_docid, open(dump_path + "adult_docid_map.pkl", "wb"))
    pickle.dump(actor_docid, open(dump_path + "actor_docid_map.pkl", "wb"))
    pickle.dump(actress_docid, open(dump_path + "actress_docid_map.pkl", "wb"))
    pickle.dump(producer_docid, open(dump_path + "producer_docid_map.pkl", "wb"))
    pickle.dump(writer_docid, open(dump_path + "writer_docid_map.pkl", "wb"))
    pickle.dump(director_docid, open(dump_path + "director_docid_map.pkl", "wb"))
    pickle.dump(composer_docid, open(dump_path + "composer_docid_map.pkl", "wb"))
    pickle.dump(cinematographer_docid, open(dump_path + "cinematographer_docid_map.pkl", "wb"))
    pickle.dump(editor_docid, open(dump_path + "editor_docid_map.pkl", "wb"))
    pickle.dump(prod_designer_docid, open(dump_path + "prod_designer_docid_map.pkl", "wb"))
    pickle.dump(dir_adult_docid, open(dump_path + "dir_adult_docid_map.pkl", "wb"))
    pickle.dump(dir_actor_docid, open(dump_path + "dir_actor_docid_map.pkl", "wb"))
    pickle.dump(dir_actress_docid, open(dump_path + "dir_actress_docid_map.pkl", "wb"))
    pickle.dump(dir_producer_docid, open(dump_path + "dir_producer_docid_map.pkl", "wb"))
    pickle.dump(dir_writer_docid, open(dump_path + "dir_writer_docid_map.pkl", "wb"))
    pickle.dump(dir_composer_docid, open(dump_path + "dir_composer_docid_map.pkl", "wb"))
    pickle.dump(dir_cinematographer_docid, open(dump_path + "dir_cinematographer_docid_map.pkl", "wb"))
    pickle.dump(dir_editor_docid, open(dump_path + "dir_editor_docid_map.pkl", "wb"))
    pickle.dump(dir_prod_designer_docid, open(dump_path + "dir_prod_designer_docid_map.pkl", "wb"))
    pickle.dump(actor_actress_docid, open(dump_path + "actor_actress_docid_map.pkl", "wb"))
