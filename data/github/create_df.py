import json
import re
import pickle
import pandas as pd


def check_null_return(t):
    if t is None or len(t) == 0:
        return ""
    return t


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?_\"\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess(input):
    return clean_str(input)


if __name__ == '__main__':
    base_path = "./"
    f = open(base_path + "AI_Hier.json", "r")
    lines = f.readlines()
    f.close()

    users = []
    texts = []
    label = []
    tags = []
    for line in lines:
        t = json.loads(line.strip())

        temp = preprocess(t["text"].strip())
        if len(t["repo_name_seg"]) == 0:
            texts.append(temp)
        else:
            texts.append(t["repo_name_seg"] + " . " + temp)

        users.append(check_null_return(t["user"]))
        label.append(check_null_return(t["sub_label"]))
        tags.append(t["tags"])

    dic = {}
    dic["user"] = users
    dic["text"] = texts
    dic["label"] = label
    dic["tags"] = tags

    df = pd.DataFrame.from_dict(dic)
    pickle.dump(df, open(base_path + "df_ai.pkl", "wb"))
    pass
