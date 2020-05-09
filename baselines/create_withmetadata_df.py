import pickle

if __name__ == "__main__":
    basepath = "../data/"
    dataset = "books/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_phrase_removed_stopwords.pkl", "rb") as handler:
        df = pickle.load(handler)

    texts = []
    for i, row in df.iterrows():
        pub = row["publisher"]
        line = row["text"]
        label = row["label"]
        year = row["publication_year"]
        auths = row["authors"]

        pub_pre = "PUB"
        auth_pre = "AUTH"
        auth_pub_pre = "AUTHPUB"
        pub_year = "PUBYEAR"

        metadata = []
        for aut in auths:
            metadata.append(auth_pre + str(aut))
            metadata.append(auth_pub_pre + str(aut) + str(pub))

        metadata.append(pub_pre + str(pub))

        metadata.append(pub_year + str(pub) + str(year))
        line = line + " " + " ".join(metadata)
        texts.append(line)

    df["text"] = texts

    pickle.dump(df, open(pkl_dump_dir + "df_phrase_removed_stopwords_baseline_metadata.pkl", "wb"))