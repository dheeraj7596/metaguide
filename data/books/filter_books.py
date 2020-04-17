import gzip
import json
import random
import pickle


def load_data(file_name, keys=None, head=500):
    count = 0
    data = []
    title_dict = {}
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)

            if d["language_code"] is None or d["language_code"] != "eng":
                continue

            flag = 0
            if keys is not None:
                for key in keys:
                    if d[key] is None or len(d[key]) == 0:
                        flag = 1
                        break

            if flag == 0:
                try:
                    temp = title_dict[d["title"].lower()]
                    continue
                except:
                    pass

                title_dict[d["title"].lower()] = 1
                count += 1
                data.append(d)

            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data


def create_author_doc_dict(books):
    author_count_dict = {}
    for i, book in enumerate(books):
        for aut in book["authors"]:
            try:
                author_count_dict[aut["author_id"]].append(i)
            except:
                author_count_dict[aut["author_id"]] = [i]
    author_count_dict = {k: v for k, v in sorted(author_count_dict.items(), key=lambda item: -len(item[1]))}
    return author_count_dict


def shortlist(books, auth_bookids, thresh=5000):
    items = list(auth_bookids.items())
    doc_id_set = set()
    selected_authors = []
    while len(doc_id_set) <= thresh:
        if len(items) == 0:
            break
        popped = items.pop(random.randrange(len(items)))
        selected_authors.append(popped[0])
        doc_id_set.update(set(popped[1]))
    filtered_books_cat = []
    for i in doc_id_set:
        filtered_books_cat.append(books[i])
    print(len(selected_authors))
    return filtered_books_cat


def remove_intersection(filtered_books, categories):
    book_id_sets = []
    title_sets = []
    for cat in categories:
        temp_set = set()
        temp_title_set = set()
        for book in filtered_books[cat]:
            temp_set.add(book["book_id"])
            temp_title_set.add(book["title"].lower())
        book_id_sets.append(temp_set)
        title_sets.append(temp_title_set)

    intersection_ids = set()
    for i, book_ids in enumerate(book_id_sets):
        for j, book_ids_in in enumerate(book_id_sets):
            if i == j:
                continue
            intersection_ids.update(book_ids.intersection(book_ids_in))

    intersection_titles = set()
    for i, titles in enumerate(title_sets):
        for j, titles_in in enumerate(title_sets):
            if i == j:
                continue
            intersection_titles.update(titles.intersection(titles_in))

    for id in intersection_ids:
        for cat in categories:
            id_list = []
            for i, book in enumerate(filtered_books[cat]):
                if book["book_id"] == id:
                    id_list.append(i)

            id_list.reverse()
            for i in id_list:
                del filtered_books[cat][i]

    for title in intersection_titles:
        for cat in categories:
            id_list = []
            for i, book in enumerate(filtered_books[cat]):
                if book["title"].lower() == title:
                    id_list.append(i)

            id_list.reverse()
            for i in id_list:
                del filtered_books[cat][i]

    return filtered_books


if __name__ == "__main__":
    base_path = "./"
    keys = ["title", "description", "authors", "publisher", "book_id", "num_pages", "publication_year"]

    categories = ["children", "young_adult", "comics_graphic", "fantasy_paranormal", "history_biography",
                  "mystery_thriller_crime", "poetry", "romance"]
    books = {}
    authors = {}
    for cat in categories:
        books[cat] = load_data(base_path + "goodreads_books_" + cat + ".json.gz", keys, head=50000)

    for cat in categories:
        authors[cat] = create_author_doc_dict(books[cat])

    filtered_authors = {}
    for cat in categories:
        filtered_authors[cat] = dict(filter(lambda entry: len(entry[1]) >= 5, authors[cat].items()))

    del authors

    filtered_books = {}
    for cat in categories:
        print("Category: ", cat)
        filtered_books[cat] = shortlist(books[cat], filtered_authors[cat])

    del books

    filtered_books = remove_intersection(filtered_books, categories)

    print("*" * 80)
    for cat in categories:
        print(len(filtered_books[cat]))

    pickle.dump(filtered_books, open(base_path + "filtered_books.pkl", "wb"))
    pass
