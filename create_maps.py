import pickle

if __name__ == "__main__":
    data_path = "./data/"
    df = pickle.load(open(data_path + "df_cs_2014_filtered.pkl", "rb"))
    label_id = {}
    id_label = {}
    labels = list(set(df.categories))
    for i, lbl in enumerate(labels):
        label_id[lbl] = i
        id_label[i] = lbl

    authors_set = set()
    for auts in df.authors:
        authors_set.update(set(auts))

    author_id = {}
    id_author = {}

    for i, aut in enumerate(authors_set):
        author_id[aut] = i
        id_author[i] = aut

    pickle.dump(label_id, open(data_path + "label_id.pkl", "wb"))
    pickle.dump(id_label, open(data_path + "id_label.pkl", "wb"))

    pickle.dump(author_id, open(data_path + "author_id.pkl", "wb"))
    pickle.dump(id_author, open(data_path + "id_author.pkl", "wb"))
