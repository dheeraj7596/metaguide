import pickle
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_path = "./"
    bus_df = pickle.load(open(base_path + "business_reviews.pkl", "rb"))

    length_list = []
    reviews_list = list(bus_df.Review)
    for r in reviews_list:
        length_list.append(len(r))

    plt.figure()
    n = plt.hist(length_list, color='blue', edgecolor='black', bins=50)
    print(n)
    plt.show()
    pass