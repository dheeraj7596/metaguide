from word2vec_orig import create_corpus, create_vocabulary
from construct_graph import detect_phrase
from data.yelp.attribute_utils import *
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
        abstract_str = row["Review"]
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
        auts = row["Users"]
        for author in auts:
            try:
                author_docid[author].add(i)
            except:
                author_docid[author] = {i}
    return author_docid


def create_attr_doc_id_map(df):
    attr_docid_map = {}
    for i, row in df.iterrows():
        row_keys = get_all_keys(row)
        for key in row_keys:
            try:
                attr_docid_map[key].add(i)
            except:
                attr_docid_map[key] = {i}
    return attr_docid_map


def create_author_attr_doc_id_map(df):
    author_attr_docid_map = {}
    for i, row in df.iterrows():
        row_keys = get_all_keys(row)
        auts = row["Users"]
        for aut in auts:
            for key in row_keys:
                try:
                    author_attr_docid_map[(aut, key)].add(i)
                except:
                    author_attr_docid_map[(aut, key)] = {i}
    return author_attr_docid_map


if __name__ == "__main__":
    data_path = "./"

    df = pickle.load(
        open(data_path + "business_1review_shortlisted_thresh_3_phrase_removed_stopwords.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))

    phrase_docid = create_phrase_doc_id_map(df, tokenizer, phrase_id_map)
    author_docid_map = create_author_doc_id_map(df)
    attr_docid_map = create_attr_doc_id_map(df)
    author_attr_docid_map = create_author_attr_doc_id_map(df)

    pickle.dump(phrase_docid, open(data_path + "phrase_docid_map_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(author_docid_map, open(data_path + "author_docid_map_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(attr_docid_map, open(data_path + "attr_docid_map_1review_shortlisted_thresh_3.pkl", "wb"))
    pickle.dump(author_attr_docid_map, open(data_path + "author_attr_docid_map_1review_shortlisted_thresh_3.pkl", "wb"))
