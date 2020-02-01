import pickle
from bs4 import BeautifulSoup

if __name__ == "__main__":
    f = open("./segmentation.txt", "r")
    lines = f.readlines()
    f.close()

    df = pickle.load(open("df_mapped_labels.pkl", "rb"))
    label_phrases = {}
    labels = list(df["label"])

    for i, line in enumerate(lines):
        line = line.lower()
        soup = BeautifulSoup(line)
        label = labels[i]
        for p in soup.findAll("phrase"):
            phrase = p.string
            try:
                temp = label_phrases[label]
                try:
                    label_phrases[label][phrase] += 1
                except:
                    label_phrases[label][phrase] = 1
            except:
                label_phrases[label] = {}
                label_phrases[label][phrase] = 1

    pass