import pickle
from nltk.corpus import stopwords
import string

if __name__ == "__main__":
    basepath = "./data/"
    dataset = "github/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_bio_phrase.pkl", "rb"))
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    abstracts = list(df["text"])

    clean_abstracts = []
    for abs in abstracts:
        word_list = abs.strip().split()
        filtered_words = [word for word in word_list if word not in stop_words]
        temp = " ".join(filtered_words)
        # temp2 = temp.translate(str.maketrans('', '', string.punctuation))
        clean_abstracts.append(temp)

    # df["Review"] = clean_abstracts
    df["text"] = clean_abstracts

    pickle.dump(df, open(pkl_dump_dir + "df_bio_phrase_removed_stopwords.pkl", "wb"))
