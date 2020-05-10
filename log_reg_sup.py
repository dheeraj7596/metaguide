import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    basepath = "./data/"
    dataset = "books/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_baseline_metadata.pkl", "rb"))
    vectorizer = TfidfVectorizer(stop_words="english")
    clf = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))

    pass
