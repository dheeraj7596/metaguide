import itertools


def get_int_adults(flag, count_dict, l, label_adult_dict, row):
    if len(label_adult_dict) > 0 and len(label_adult_dict[l]) > 0:
        seed_adults = set(label_adult_dict[l].keys())
        int_adults = {row["isAdult"]}.intersection(seed_adults)
    else:
        int_adults = []

    for adult in int_adults:
        try:
            temp = count_dict[l]
        except:
            count_dict[l] = {}
        count_dict[l]["ADULT_" + str(adult)] = label_adult_dict[l][adult]
        flag = 1
    return count_dict, flag


def get_int_unigram(flag, count_dict, l, label_entity_dict, row, col):
    if len(label_entity_dict) > 0 and len(label_entity_dict[l]) > 0:
        seed_entities = set(label_entity_dict[l].keys())
        int_entities = set(row[col]).intersection(seed_entities)
    else:
        int_entities = []

    for ent in int_entities:
        try:
            temp = count_dict[l]
        except:
            count_dict[l] = {}
        count_dict[l][col + str(ent)] = label_entity_dict[l][ent]
        flag = 1
    return count_dict, flag


def get_int_dir_adult(flag, count_dict, l, label_dir_adult_dict, row):
    if len(label_dir_adult_dict) > 0 and len(label_dir_adult_dict[l]) > 0:
        seed_dir_adults = set(label_dir_adult_dict[l].keys())
        col1_entities = row["director"]
        col2_entities = {row["isAdult"]}
        cols_set = set(itertools.product(col1_entities, col2_entities))
        int_dir_adults = cols_set.intersection(seed_dir_adults)
    else:
        int_dir_adults = []

    for ent in int_dir_adults:
        try:
            temp = count_dict[l]
        except:
            count_dict[l] = {}
        count_dict[l]["Dir_Adult" + str(ent)] = label_dir_adult_dict[l][ent]
        flag = 1
    return count_dict, flag


def get_int_bigram(flag, count_dict, l, label_bigram_entity_dict, row, col1, col2):
    if len(label_bigram_entity_dict) > 0 and len(label_bigram_entity_dict[l]) > 0:
        seed_bigram_entities = set(label_bigram_entity_dict[l].keys())
        col1_entities = row[col1]
        col2_entities = row[col2]
        cols_set = set(itertools.product(col1_entities, col2_entities))
        int_bigram_entities = cols_set.intersection(seed_bigram_entities)
    else:
        int_bigram_entities = []

    for ent in int_bigram_entities:
        try:
            temp = count_dict[l]
        except:
            count_dict[l] = {}
        count_dict[l][col1 + "_" + col2 + str(ent)] = label_bigram_entity_dict[l][ent]
        flag = 1
    return count_dict, flag
