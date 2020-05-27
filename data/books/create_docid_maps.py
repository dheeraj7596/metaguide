from word2vec_orig import create_corpus, create_vocabulary
from construct_graph import detect_phrase
import pickle


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


def create_author_doc_id_map(df):
    author_docid = {}
    for i, row in df.iterrows():
        auts = row["authors"]
        for author in auts:
            try:
                author_docid[author].add(i)
            except:
                author_docid[author] = {i}
    return author_docid


def create_pub_doc_id_map(df):
    pub_docid_map = {}
    for i, row in df.iterrows():
        pub = row["publisher"]
        try:
            pub_docid_map[pub].add(i)
        except:
            pub_docid_map[pub] = {i}
    return pub_docid_map


def create_year_doc_id_map(df):
    year_docid_map = {}
    for i, row in df.iterrows():
        year = row["publication_year"]
        try:
            year_docid_map[year].add(i)
        except:
            year_docid_map[year] = {i}
    return year_docid_map


def create_author_pub_doc_id_map(df):
    author_pub_docid_map = {}
    for i, row in df.iterrows():
        auts = row["authors"]
        pub = row["publisher"]
        for aut in auts:
            try:
                author_pub_docid_map[(aut, pub)].add(i)
            except:
                author_pub_docid_map[(aut, pub)] = {i}
    return author_pub_docid_map


def create_author_year_doc_id_map(df):
    author_year_docid_map = {}
    for i, row in df.iterrows():
        auts = row["authors"]
        pub = row["publication_year"]
        for aut in auts:
            try:
                author_year_docid_map[(aut, pub)].add(i)
            except:
                author_year_docid_map[(aut, pub)] = {i}
    return author_year_docid_map


def create_pub_year_doc_id_map(df):
    pub_year_docid_map = {}
    for i, row in df.iterrows():
        year = row["publication_year"]
        pub = row["publisher"]
        try:
            pub_year_docid_map[(pub, year)].add(i)
        except:
            pub_year_docid_map[(pub, year)] = {i}
    return pub_year_docid_map


if __name__ == "__main__":
    data_path = "./"

    df = pickle.load(open(data_path + "df_phrase_removed_stopwords.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))

    phrase_docid = create_phrase_doc_id_map(df, tokenizer, phrase_id_map)
    author_docid_map = create_author_doc_id_map(df)
    pub_docid_map = create_pub_doc_id_map(df)
    year_docid_map = create_year_doc_id_map(df)
    author_pub_docid_map = create_author_pub_doc_id_map(df)
    author_year_docid_map = create_author_year_doc_id_map(df)
    pub_year_docid_map = create_pub_year_doc_id_map(df)

    pickle.dump(phrase_docid, open(data_path + "phrase_docid_map.pkl", "wb"))
    pickle.dump(author_docid_map, open(data_path + "author_docid_map.pkl", "wb"))
    pickle.dump(pub_docid_map, open(data_path + "pub_docid_map.pkl", "wb"))
    pickle.dump(year_docid_map, open(data_path + "year_docid_map.pkl", "wb"))
    pickle.dump(author_pub_docid_map, open(data_path + "author_pub_docid_map.pkl", "wb"))
    pickle.dump(pub_year_docid_map, open(data_path + "pub_year_docid_map.pkl", "wb"))
    pickle.dump(author_year_docid_map, open(data_path + "author_year_docid_map.pkl", "wb"))
