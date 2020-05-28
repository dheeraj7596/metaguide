import pickle

if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "df_summary_top6_title_summary_all_reviews.pkl", "rb"))

    reviews = list(df["text"])
    mod_reviews = []
    for i, r in enumerate(reviews):
        encoded_string = r.encode("ascii", "ignore")
        decode_string = encoded_string.decode()
        mod_reviews.append(decode_string)
    df["text"] = mod_reviews
    pickle.dump(df, open(base_path + "df_summary_top6_title_summary_all_reviews_modified.pkl", "wb"))

    # reviews = list(df["text"])
    #
    # f = open("./text.txt", "w")
    # for i, r in enumerate(reviews):
    #     encoded_string = r.encode("ascii", "ignore")
    #     decode_string = encoded_string.decode()
    #     f.write(decode_string)
    #     f.write("\n")
    # f.close()

    # f2 = open("./text_title_sum.txt", "w")
    # for i, row in df.iterrows():
    #     title = row["aka_title"].lower()
    #     summary = row["summary"].lower()
    #     temp = title + " . " + summary
    #     f2.write(temp)
    #     f2.write("\n")
    #
    # f2.close()
