import pickle
import pandas as pd
import random
from data.yelp.create_business_data import remove_b_prefix


def compute_business_id_to_review_index(reviews):
    business_id_to_review_id = {}
    user_id_to_business_ids = {}
    for i, row in reviews.iterrows():
        bus_id = row["business_id"]
        user_id = row["user_id"]
        try:
            business_id_to_review_id[bus_id].append(i)
        except:
            business_id_to_review_id[bus_id] = [i]
        try:
            user_id_to_business_ids[user_id].append(bus_id)
        except:
            user_id_to_business_ids[user_id] = [bus_id]
    return business_id_to_review_id, user_id_to_business_ids


def random_n(arr, k):
    random.shuffle(arr)
    return arr[:k]


def get_reviews_users(reviews, business_id_to_review_index):
    business_id_to_reviews = {}
    business_id_to_userids = {}

    for bus_id in business_id_to_review_index:
        review_ids = business_id_to_review_index[bus_id]
        rows = reviews.ix[review_ids]
        concat_review = ""
        user_list = []
        for i, row in rows.iterrows():
            if len(concat_review) == 0:
                concat_review = remove_b_prefix(row["text"]).lower()
            else:
                concat_review = concat_review + " " + remove_b_prefix(row["text"]).lower()
            user_list.append(remove_b_prefix(row["user_id"]))

        business_id_to_reviews[bus_id] = concat_review
        business_id_to_userids[bus_id] = user_list
    return business_id_to_reviews, business_id_to_userids


def filter_selected_reviews(reviews, business_id_to_review_index):
    indices = set()
    for b in business_id_to_review_index:
        indices.update(set(business_id_to_review_index[b]))
    filtered_reviews = reviews.ix[list(indices)]
    filtered_reviews = filtered_reviews.reset_index(drop=True)
    return filtered_reviews


if __name__ == "__main__":
    base_path = "./"
    bus_df = pickle.load(
        open(base_path + "business_reviews_phrase_removed_stopwords_labeled_shortlisted_thresh_3.pkl", "rb"))
    reviews = pd.read_csv(base_path + "review.csv")
    bus_ids = list(bus_df["business_id"])
    reviews = reviews[reviews.business_id.isin(bus_ids)]
    reviews = reviews[reviews['text'].notna()]
    reviews = reviews.reset_index(drop=True)

    business_id_to_review_index, user_id_to_business_ids = compute_business_id_to_review_index(reviews)
    for bus_id in list(business_id_to_review_index.keys()):
        random_review_ids = random_n(business_id_to_review_index[bus_id], 1)
        business_id_to_review_index[bus_id] = random_review_ids

    business_id_to_reviews, business_id_to_userids = get_reviews_users(reviews, business_id_to_review_index)

    reviews_list = []
    combined_users_list = []
    for i, row in bus_df.iterrows():
        business_id = row["business_id"]
        try:
            reviews_list.append(business_id_to_reviews[business_id])
            combined_users_list.append(business_id_to_userids[business_id])
        except:
            print("Business id not found: ", business_id)

    bus_df["Review"] = reviews_list
    bus_df["text"] = reviews_list
    pass
    # for bus_id in list(business_id_to_review_index.keys()):
    #     random_review_ids = random_n(business_id_to_review_index[bus_id], 1)
    #     business_id_to_review_index[bus_id] = random_review_ids
    #
    # business_id_to_reviews, business_id_to_userids = get_reviews_users(reviews, business_id_to_review_index)
    #
    # reviews_list = []
    # combined_users_list = []
    # for i, row in bus_df.iterrows():
    #     business_id = row["business_id"]
    #     try:
    #         reviews_list.append(business_id_to_reviews[business_id])
    #         combined_users_list.append(business_id_to_userids[business_id])
    #     except:
    #         print("Business id not found: ", business_id)
    #
    # bus_df["Review"] = reviews_list
    # bus_df["Users"] = combined_users_list
    #
    # selected_reviews = filter_selected_reviews(reviews, business_id_to_review_index)

    # bus_df.to_csv(base_path + "business_reviews.csv")
    # pickle.dump(bus_df, open(base_path + "business_reviews.pkl", "wb"))
    # pickle.dump(selected_reviews, open(base_path + "selected_reviews.pkl", "wb"))
