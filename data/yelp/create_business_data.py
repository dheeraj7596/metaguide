import pickle
import pandas as pd
import json


def my_filter(row):
    categories = remove_b_prefix(row["categories"]).split(",")
    categories = [cat.strip().lower() for cat in categories]
    if len(set(categories).intersection(main_categories_filter)) > 0:
        return True
    return False


def fine_filter(row):
    categories = remove_b_prefix(row["categories"]).split(",")
    categories = [cat.strip().lower() for cat in categories]
    if len(set(categories).intersection(fine_grained_categories)) == 1:
        return True
    return False


def add_b_prefix(str):
    return "b\'" + str + "\'"


def remove_b_prefix(str):
    return str[2:-1]


def print_categories(df):
    cat_dict = {}
    for i, row in df.iterrows():
        categories = remove_b_prefix(row["categories"]).split(",")
        categories = [cat.strip().lower() for cat in categories]
        for cat in categories:
            try:
                cat_dict[cat] += 1
            except:
                cat_dict[cat] = 1

    print(json.dumps(cat_dict, indent=4))


def top_categories(k):
    with open(base_path + "categories.json") as json_file:
        cat_dic = json.load(json_file)
    items = cat_dic.items()
    items_sorted = sorted(items, key=lambda item: -item[1])
    cat_dic_sorted = dict(items_sorted[:15])
    print(cat_dic_sorted)
    return list(cat_dic_sorted.keys())


if __name__ == "__main__":
    base_path = "./"
    k = 15
    main_categories_filter = {"restaurants", "food"}

    bus_df = pd.read_csv(base_path + "business.csv")
    bus_df = bus_df[bus_df['categories'].notna()]

    bus_df = bus_df.reset_index(drop=True)

    bus_df = bus_df[bus_df.apply(my_filter, axis=1)]
    bus_df = bus_df[bus_df["review_count"] >= 10]
    bus_df = bus_df.reset_index(drop=True)
    fine_grained_categories = top_categories(k)

    bus_df = bus_df[bus_df.apply(fine_filter, axis=1)]
    bus_df = bus_df.reset_index(drop=True)
    print(len(bus_df))

    labels = []
    for i, row in bus_df.iterrows():
        categories = remove_b_prefix(row["categories"]).split(",")
        categories = [cat.strip().lower() for cat in categories]
        label = list(set(categories).intersection(fine_grained_categories))[0]
        labels.append(label)

    bus_df["label"] = labels
    pickle.dump(bus_df, open(base_path + "business_shortlisted.pkl", "wb"))
