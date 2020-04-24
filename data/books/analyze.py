import pickle
import matplotlib.pyplot as plt


def analyze_authors():
    author_count_dict = {}
    for i, row in df.iterrows():
        auths = row["authors"]
        for aut in auths:
            try:
                author_count_dict[aut] += 1
            except:
                author_count_dict[aut] = 1
    plt.figure()
    n = plt.hist(list(author_count_dict.values()), color='blue', edgecolor='black', bins=100)
    plt.show()
    label_author_dict = {}
    for i, row in df.iterrows():
        l = row["label"]
        try:
            label_author_dict[l].update(set(row["authors"]))
        except:
            label_author_dict[l] = set(row["authors"])
    labels = list(label_author_dict.keys())
    intersection_authors = set()
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            if l1 == l2:
                continue
            intersection_authors.update(label_author_dict[l1].intersection(label_author_dict[l2]))
    print(len(intersection_authors))


def analyze_pub():
    print("Number of distinct publishers: ", len(set(df["publisher"])))
    pub_count_dict = {}
    for i, row in df.iterrows():
        pub = row["publisher"]
        try:
            pub_count_dict[pub] += 1
        except:
            pub_count_dict[pub] = 1
    plt.figure()
    n = plt.hist(list(pub_count_dict.values()), color='blue', edgecolor='black', bins=1000)
    plt.show()

    label_pub_dict = {}
    for i, row in df.iterrows():
        l = row["label"]
        try:
            label_pub_dict[l].add(row["publisher"])
        except:
            label_pub_dict[l] = {row["publisher"]}
    labels = list(label_pub_dict.keys())
    intersection_pubs = set()
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            if l1 == l2:
                continue
            intersection_pubs.update(label_pub_dict[l1].intersection(label_pub_dict[l2]))
    print(len(intersection_pubs))


if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "df_phrase_removed_stopwords.pkl", "rb"))

    # analyze_authors()
    analyze_pub()
    pass
