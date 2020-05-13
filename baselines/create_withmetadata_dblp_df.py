import pickle
import itertools

if __name__ == "__main__":
    basepath = "../data/"
    dataset = "dblp/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_mapped_labels_phrase_removed_stopwords.pkl", "rb") as handler:
        df = pickle.load(handler)

    texts = []
    for i, row in df.iterrows():
        auths = row["authors"].split(",")
        line = row["abstract"]
        label = row["label"]
        year = row["year"]
        author_pairs = list(itertools.combinations(auths, 2))

        auth_pre = "AUTH"
        auth_pairs_pre = "AUTHPAIR"
        pub_year_pre = "YEAR"
        auth_year_pre = "AUTHYEAR"

        metadata = []
        for aut in auths:
            metadata.append((auth_pre + str(aut)).replace(" ", ""))
            metadata.append((auth_year_pre + str(aut) + str(year)).replace(" ", ""))

        metadata.append(pub_year_pre + str(year))

        for p in author_pairs:
            metadata.append((auth_pairs_pre + p[0] + p[1]).replace(" ", ""))

        line = line + " " + " ".join(metadata)
        texts.append(line)

    df["abstract"] = texts

    pickle.dump(df, open(pkl_dump_dir + "df_mapped_labels_phrase_removed_stopwords_baseline_metadata.pkl", "wb"))
