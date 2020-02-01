import pickle


if __name__ == "__main__":
    df = pickle.load(open("./df_all.pkl", "rb"))
    abs_df = df[df.abstract.notnull()]
    conf_abs_df = abs_df[abs_df.conf.notnull()]
    conf_abs_df = conf_abs_df[conf_abs_df.authors.notnull()]
    pass