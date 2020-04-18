import pickle
import pandas as pd


def get_authors(auth_list):
    temp_list = []
    for aut in auth_list:
        temp_list.append(aut["author_id"])
    return temp_list


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
            authors_list = get_authors(book["authors"])
            final_dic["authors"].append(authors_list)
            final_dic["publisher"].append(book["publisher"])
            final_dic["book_id"].append(book["book_id"])
            final_dic["num_pages"].append(book["num_pages"])
            final_dic["publication_year"].append(book["publication_year"])

            text = book["title"].lower() + " . " + book["description"].lower()
            arr = text.splitlines()
            final_dic["text"].append(" ".join(arr))

    df = pd.DataFrame.from_dict(final_dic)
    pickle.dump(df, open(base_path + "df.pkl", 'wb'))

    pass
