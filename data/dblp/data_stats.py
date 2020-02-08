import pickle

if __name__ == "__main__":
    df = pickle.load(open("./df_mapped_labels_phrase_removed_stopwords.pkl", "rb"))

    auth_dict = {}
    labels = set(df["label"])
    for l in labels:
        auth_dict[l] = {}

    for i, row in df.iterrows():
        authors = row["authors"].split(",")
        label = row["label"]
        for aut in authors:
            try:
                auth_dict[label][aut] += 1
            except:
                auth_dict[label][aut] = 1

    for l in labels:
        auth_dict[l] = {k: v for k, v in sorted(auth_dict[l].items(), key=lambda item: -item[1])}

    f = open("./authors.txt", "w")
    for l in auth_dict:
        f.write(l)
        f.write("\n")
        for au in auth_dict[l]:
            f.write(au + " : " + str(auth_dict[l][au]))
            f.write("\n")
        f.write("******************************************************\n")

    f.close()
    pass