import pickle

if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "df_ai.pkl", "rb"))

    reviews = list(df["text"])

    f = open("./text.txt", "w")
    for i, r in enumerate(reviews):
        f.write(r)
        f.write("\n")

    f.close()
