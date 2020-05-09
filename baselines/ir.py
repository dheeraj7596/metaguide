from coc_data_utils import get_label_term_json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import classification_report
from cocube_beta import modify_phrases

if __name__ == "__main__":
    basepath = "../data/"
    dataset = "books/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "df_phrase_removed_stopwords_baseline_metadata.pkl", "rb") as handler:
        df = pickle.load(handler)

    phrase_id_map = pickle.load(open(pkl_dump_dir + "phrase_id_map.pkl", "rb"))

    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords.json")
    label_term_dict = modify_phrases(label_term_dict, phrase_id_map)

    print(label_term_dict)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    X_arr = X.toarray()
    names = vectorizer.get_feature_names()

    label_term_index_dict = {}
    for i in label_term_dict:
        label_term_index_dict[i] = []
        for w in label_term_dict[i]:
            try:
                label_term_index_dict[i].append(names.index(w))
            except Exception as e:
                print("Exception for: ", w, e)

    pred = []
    for i in X_arr:
        maxi = -1
        max_l = ""
        for l in label_term_index_dict:
            sum = 0
            for ind in label_term_index_dict[l]:
                sum += i[ind]
            if sum > maxi:
                maxi = sum
                max_l = l

        pred.append(max_l)

    print(classification_report(df["label"], pred))
