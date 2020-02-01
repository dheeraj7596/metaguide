from bs4 import BeautifulSoup
import bleach
import pickle


def generate_name(id):
    return "fnust" + str(id)


def decrypt(val):
    try:
        if val[:5] == "fnust":
            return int(val[5:])
    except:
        return None


if __name__ == "__main__":
    base_path = "./data/"
    dataset = "dblp/"
    data_path = base_path + dataset
    out_path = data_path + "segmentation.txt"
    df = pickle.load(open(data_path + "df_mapped_labels.pkl", "rb"))
    f = open(out_path, "r")
    lines = f.readlines()
    f.close()

    phrase_id_map = {}
    counter = 0
    data = []

    for line in lines:
        line = line.lower()
        soup = BeautifulSoup(line)
        for p in soup.findAll("phrase"):
            phrase = p.string
            try:
                temp = phrase_id_map[phrase]
            except:
                phrase_id_map[phrase] = counter
                counter += 1
            name = generate_name(phrase_id_map[phrase])
            p.string.replace_with(name)
        temp_str = bleach.clean(str(soup), tags=[], strip=True)
        data.append(temp_str)

    df["abstract"] = data
    pickle.dump(df, open(data_path + "df_mapped_labels_phrase.pkl", "wb"))
    pickle.dump(phrase_id_map, open(data_path + "phrase_id_map.pkl", "wb"))
