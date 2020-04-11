import pickle

if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "business_1review_shortlisted_thresh_3.pkl", "rb"))

    reviews = list(df["Review"])

    f = open("./reviews.txt", "w")
    for i, r in enumerate(reviews):
        f.write(r)
        f.write("\n")

    f.close()
