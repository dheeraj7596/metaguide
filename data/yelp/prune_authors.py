import pickle


def create_count_dict(auth_count_dict):
    count_dict = {}
    for auth in auth_count_dict:
        try:
            count_dict[auth_count_dict[auth]] += 1
        except:
            count_dict[auth_count_dict[auth]] = 1
    return count_dict


def create_auth_count_dict(authors_list):
    auth_count_dict = {}
    for authors in authors_list:
        for aut in authors:
            try:
                auth_count_dict[aut.strip()] += 1
            except:
                auth_count_dict[aut.strip()] = 1
    return auth_count_dict


def filter_authors(authors_list, auth_count_dict, threshold=3):
    empty_count = 0
    filtered_authors_list = []
    for authors in authors_list:
        temp_authors = []
        for aut in authors:
            aut = aut.strip()
            if auth_count_dict[aut] >= threshold:
                temp_authors.append(aut)
        if len(temp_authors) == 0:
            empty_count += 1
        filtered_authors_list.append(temp_authors)
    print("Empty count: ", empty_count)
    return filtered_authors_list


if __name__ == "__main__":
    pkl_dump_dir = "./"
    df = pickle.load(open(pkl_dump_dir + "business_reviews_phrase_removed_stopwords_labeled.pkl", "rb"))
    print("Length of dataframe: ", len(df))
    authors_list = list(df.Users)
    diff_authors = set()
    for authors in authors_list:
        diff_authors.update(set(authors))
    print("Total number of distinct authors: ", len(diff_authors))
    auth_count_dict = create_auth_count_dict(authors_list)
    count_dict = create_count_dict(auth_count_dict)
    print(dict(sorted(count_dict.items(), key=lambda x: x[0])))

    filtered_authors = filter_authors(authors_list, auth_count_dict, threshold=5)
    df["Users"] = filtered_authors
    pickle.dump(df, open(pkl_dump_dir + "business_reviews_phrase_removed_stopwords_labeled_thresh_5.pkl", "wb"))
    pass
