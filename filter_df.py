import pandas as pd
import pickle

if __name__ == "__main__":
    data_path = "./data/"
    df = pickle.load(open(data_path + "df_cs_2014.pkl", "rb"))
    filtered_labels = ["cs.cv",
                       "cs.cl",
                       "cs.ni",
                       "cs.cr",
                       "cs.ds",
                       "cs.ai",
                       "cs.dc",
                       "cs.sy",
                       "cs.se",
                       "cs.ro",
                       "cs.lo",
                       "cs.lg",
                       "cs.cy",
                       "cs.gt",
                       "cs.db",
                       "cs.hc",
                       "cs.si",
                       "cs.pl",
                       "cs.ir",
                       "cs.cg",
                       "cs.ne",
                       "cs.cc",
                       "cs.dl",
                       "cs.dm",
                       "cs.fl",
                       "cs.oh"]
    df_new = df[df.categories.isin(filtered_labels)]
    df_new = df_new.reset_index(drop=True)
