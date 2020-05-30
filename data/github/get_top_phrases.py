import pickle
from bs4 import BeautifulSoup
import json

if __name__ == "__main__":
    f = open("./segmentation.txt", "r")
    lines = f.readlines()
    f.close()

    df = pickle.load(open("df_ai.pkl", "rb"))
    label_phrases = {}
    labels = list(df["label"])

    for i, line in enumerate(lines):
        line = line.lower()
        soup = BeautifulSoup(line)
        label = labels[i]
        for p in soup.findAll("phrase"):
            phrase = p.string
            if phrase is None:
                continue
            try:
                temp = label_phrases[label]
                try:
                    label_phrases[label][phrase] += 1
                except:
                    label_phrases[label][phrase] = 1
            except:
                label_phrases[label] = {}
                label_phrases[label][phrase] = 1

    for l in label_phrases:
        label_phrases[l] = {k: v for k, v in sorted(label_phrases[l].items(), key=lambda item: -item[1])}

    with open('./phrases_ai.json', 'w') as fp:
        json.dump(label_phrases, fp)
