import pickle

if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "df_summary_top6_all_reviews.pkl", "rb"))
    texts = []
    for i, row in df.iterrows():
        title = row["aka_title"].lower()
        summary = row["summary"].lower()
        review = row["review"].lower().replace('\n', ' ')
        temp = title + " . " + summary + " . " + review
        texts.append(temp)
    df["text"] = texts
    pickle.dump(df, open(base_path + "df_summary_top6_title_summary_all_reviews.pkl", "wb"))
    pass

# supervised
# write_to_file
# detect phrases
# parse phrases
# remove_stop_words

# construct_graph
# construct_docid_maps
# cocube
