import pickle

if __name__ == "__main__":
    data_path = "./"
    df = pickle.load(open(data_path + "business_reviews.pkl", "rb"))
    threshold = 1500
    texts = []
    for i, row in df.iterrows():
        review = row["Review"]
        temp = review[:1500]
        ind = temp.rfind(" ")
        texts.append(temp[:ind])
    df["Review"] = texts
    pickle.dump(df, open(data_path + "business_reviews_cut.pkl", "wb"))
    pass
