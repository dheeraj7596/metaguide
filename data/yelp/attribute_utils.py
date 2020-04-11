import numpy as np
import pandas as pd
import ast


def remove_b_prefix(str):
    return str[2:-1]


def get_ambience(row):
    keys = []
    prefix = "ambience"
    str = row["attributes.Ambience"]
    if pd.isna(str) or pd.isnull(str):
        return keys

    str = remove_b_prefix(str)
    json = ast.literal_eval(str)
    if json is None:
        return keys
    for key in json:
        if json[key]:
            ret_key = prefix + "_" + key + "_" + "True"
        else:
            ret_key = prefix + "_" + key + "_" + "False"
        keys.append(ret_key)
    return keys


def get_True_False_key(row, prefix, attr):
    str = row[attr]
    if pd.isna(str) or pd.isnull(str):
        return ""
    str = remove_b_prefix(str)
    if str in ["True", "False"]:
        return prefix + "_" + str
    else:
        return ""


def get_noiselevel(row):
    prefix = "noiselevel"
    str = row["attributes.NoiseLevel"]
    if pd.isna(str) or pd.isnull(str):
        return ""
    str = remove_b_prefix(str)
    if "average" in str:
        return prefix + "_average"
    elif "quiet" in str:
        return prefix + "_quiet"
    elif "loud" in str and len(str) <= 7:
        return prefix + "_loud"
    elif "very_loud" in str:
        return prefix + "_very_loud"
    else:
        return ""


def get_alcohol(row):
    prefix = "alcohol"
    str = row["attributes.Alcohol"]
    if pd.isna(str) or pd.isnull(str):
        return ""
    str = remove_b_prefix(str)
    if "full_bar" in str:
        return prefix + "_full_bar"
    elif "beer_and_wine" in str:
        return prefix + "_beer_and_wine"
    else:
        return ""


def get_good_for_meal(row):
    keys = []
    prefix = "GoodForMeal"
    str = row["attributes.GoodForMeal"]
    if pd.isna(str) or pd.isnull(str):
        return keys

    str = remove_b_prefix(str)
    json = ast.literal_eval(str)
    if json is None:
        return keys
    for key in json:
        if json[key]:
            ret_key = prefix + "_" + key + "_" + "True"
        else:
            ret_key = prefix + "_" + key + "_" + "False"
        keys.append(ret_key)
    return keys


def get_dietary_restrictions(row):
    keys = []
    prefix = "DietaryRestrictions"
    str = row["attributes.DietaryRestrictions"]
    if pd.isna(str) or pd.isnull(str):
        return keys

    str = remove_b_prefix(str)
    json = ast.literal_eval(str)
    if json is None:
        return keys
    for key in json:
        if json[key]:
            ret_key = prefix + "_" + key + "_" + "True"
        else:
            ret_key = prefix + "_" + key + "_" + "False"
        keys.append(ret_key)
    return keys


def get_restaurant_attire(row):
    prefix = "RestaurantsAttire"
    str = row["attributes.RestaurantsAttire"]
    if pd.isna(str) or pd.isnull(str):
        return ""
    str = remove_b_prefix(str)
    if "casual" in str:
        return prefix + "_casual"
    elif "dressy" in str:
        return prefix + "_dressy"
    elif "formal" in str:
        return prefix + "_formal"
    else:
        return ""


def get_smoking(row):
    prefix = "smoking"
    str = row["attributes.Smoking"]
    if pd.isna(str) or pd.isnull(str):
        return ""
    str = remove_b_prefix(str)
    if "no" in str:
        return prefix + "_no"
    elif "outdoor" in str:
        return prefix + "_outdoor"
    elif "yes" in str:
        return prefix + "_yes"
    else:
        return ""


def get_all_keys(row):
    true_false_attrs = [
        ("attributes.GoodForDancing", "GoodForDancing"),
        ("attributes.OutdoorSeating", "OutdoorSeating"),
        ("attributes.Corkage", "Corkage"),
        ("attributes.BYOB", "BYOB"),
        ("attributes.RestaurantsGoodForGroups", "RestaurantsGoodForGroups"),
        ("attributes.Caters", "Caters"),
        ("attributes.DriveThru", "DriveThru"),
        ("attributes.RestaurantsTakeOut", "RestaurantsTakeOut"),
        ("attributes.RestaurantsDelivery", "RestaurantsDelivery"),
        ("attributes.GoodForKids", "GoodForKids")
    ]
    total_keys = set()
    temp_keys = get_ambience(row)
    total_keys.update(set(temp_keys))

    temp_keys = get_good_for_meal(row)
    total_keys.update(set(temp_keys))

    temp_keys = get_dietary_restrictions(row)
    total_keys.update(set(temp_keys))

    temp_key = get_noiselevel(row)
    if len(temp_key) > 0:
        total_keys.add(temp_key)

    temp_key = get_alcohol(row)
    if len(temp_key) > 0:
        total_keys.add(temp_key)

    temp_key = get_restaurant_attire(row)
    if len(temp_key) > 0:
        total_keys.add(temp_key)

    temp_key = get_smoking(row)
    if len(temp_key) > 0:
        total_keys.add(temp_key)

    for attr in true_false_attrs:
        temp_key = get_True_False_key(row, attr[1], attr[0])
        if len(temp_key) > 0:
            total_keys.add(temp_key)

    return total_keys
