from word2vec_orig import create_corpus, create_vocabulary
import pickle

if __name__ == "__main__":
    base_path = "./data/"
    dataset = "dblp/"
    data_path = base_path + dataset

    df = pickle.load(open(data_path + "df_mapped_labels_phrase_removed_stopwords_test.pkl", "rb"))
    corpus = create_corpus(df)
    vocabulary, vocab_to_int, int_to_vocab, tokenizer = create_vocabulary(corpus, num_words=50000)
    pickle.dump(tokenizer, open(data_path + "tokenizer_test.pkl", "wb"))
