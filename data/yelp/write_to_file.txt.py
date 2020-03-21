import pickle

if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "business_reviews.pkl", "rb"))

    reviews = list(df["Review"])

    f = open("./reviews.txt", "w")
    for i, r in enumerate(reviews):
        f.write(r)
        f.write("\n")

    f.close()
