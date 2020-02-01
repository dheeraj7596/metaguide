import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    f = open("./data.txt", "r")
    lines = f.readlines()
    f.close()

    data_dict = {}
    cols = ["title", "authors", "year", "conf", "abstract"]
    for col in cols:
        data_dict[col] = []

    temp = {}
    for line in lines:
        if line == "\n":
            for col in cols:
                try:
                    if len(temp[col]) > 0:
                        data_dict[col].append(temp[col])
                    else:
                        data_dict[col].append(np.nan)
                except:
                    data_dict[col].append(np.nan)
            temp = {}
            continue
        line = line.strip()
        if line.startswith("#*"):
            temp["title"] = line[2:]
        elif line.startswith("#@"):
            temp["authors"] = line[2:]
        elif line.startswith("#year"):
            temp["year"] = line[5:]
        elif line.startswith("#conf"):
            temp["conf"] = line[5:]
        elif line.startswith("#!"):
            temp["abstract"] = line[2:]

    df = pd.DataFrame(data_dict)
    pickle.dump(df, open("./df_all.pkl", "wb"))
    pass
