import pickle
import pandas as pd

if __name__ == "__main__":
    base_path = "./"

    filtered_books = pickle.load(open(base_path + "filtered_books.pkl", "rb"))
    final_dic = {}
    final_dic["label"] = []
    final_dic["title"] = []
    final_dic["description"] = []
    final_dic["text"] = []
    final_dic["authors"] = []
    final_dic["publisher"] = []
    final_dic["book_id"] = []
    final_dic["num_pages"] = []
    final_dic["publication_year"] = []

    for cat in filtered_books:
        for book in filtered_books[cat]:
            final_dic["label"].append(cat)
            final_dic["title"].append(book["title"])
            final_dic["description"].append(book["description"])
            final_dic["authors"].append(book["authors"])
            final_dic["publisher"].append(book["publisher"])
            final_dic["book_id"].append(book["book_id"])
            final_dic["num_pages"].append(book["num_pages"])
            final_dic["publication_year"].append(book["publication_year"])

            text = book["title"] + " " + book["description"]
            final_dic["text"].append(text)

    df = pd.DataFrame.from_dict(final_dic)
    pickle.dump(df, open(base_path + "df.pkl", 'wb'))

    pass
