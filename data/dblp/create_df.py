import pickle


def get_json(path):
    import json
    dic = json.load(open(path, "r"))
    return dic


if __name__ == "__main__":
    conf_abs_df = pickle.load(open("./conf_abs_df.pkl", "rb"))
    dic = get_json("./mapping.json")
    temp_df = conf_abs_df[conf_abs_df["conf"].isin(list(dic.keys()))]
    mod_cols = []
    for i, row in temp_df.iterrows():
        mod_cols.append(dic[row["conf"]])
    temp_df["label"] = mod_cols
    pickle.dump(temp_df, open("./df_mapped_labels.pkl", "wb"))
    pass
