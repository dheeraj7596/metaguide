import pickle

if __name__ == "__main__":
    data_path = "./"
    df = pickle.load(open(data_path + "business_reviews_phrase_removed_stopwords_labeled.pkl", "rb"))
    labels = ["american", "chinese", "indian", "italian", "japanese", "korean", "mexican", "thai", "vietnamese"]
    temp = df[df.label.isin(labels)]
    temp = temp.reset_index(drop=True)
    pickle.dump(temp, open(data_path + "business_reviews_phrase_removed_stopwords_labeled_shortlisted.pkl", "wb"))
    pass
