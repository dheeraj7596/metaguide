from coc_data_utils import get_label_term_json
import pickle
from cocube_beta import modify_phrases
import os

if __name__ == "__main__":
    basepath = "../data/"
    dataset = "books/"
    pkl_dump_dir = basepath + dataset
    west_dump_dir = "/Users/dheerajmekala/Work/WeSTClass-master/" + dataset
    os.makedirs(west_dump_dir, exist_ok=True)

    phrase_id_map = pickle.load(open(pkl_dump_dir + "phrase_id_map.pkl", "rb"))

    df = pickle.load(open(pkl_dump_dir + "/df_baseline_metadata.pkl", "rb"))
    label_term_dict = get_label_term_json(pkl_dump_dir + "seedwords.json")
    label_term_dict = modify_phrases(label_term_dict, phrase_id_map)
    f1 = open(west_dump_dir + "classes.txt", "w")
    f2 = open(west_dump_dir + "keywords.txt", "w")

    i = 0
    for l in label_term_dict:
        f1.write(str(i) + ":" + l + "\n")
        temp = []
        for p in label_term_dict[l]:
            temp.append(p.split("$")[0])
        f2.write(str(i) + ":" + ",".join(temp) + "\n")
        i += 1
    f1.close()
    f2.close()
    df.to_csv(west_dump_dir + "dataset.csv")
