import pickle
import pandas as pd


def add(dic, title, id):
    try:
        dic[title].add(id)
    except:
        dic[title] = {id}
    return dic


def update(lis, title, dic):
    try:
        lis.append(dic[title])
    except:
        lis.append(set([]))
    return lis


def get_from_crew(crew):
    title_to_writer = {}
    title_to_director = {}

    for i, row in crew.iterrows():
        title = row["tconst"]
        title_to_director[title] = set(row["directors"].strip().split(","))
        title_to_writer[title] = set(row["writers"].strip().split(","))

    return title_to_writer, title_to_director


def get_from_principals(principals):
    title_to_actor = {}
    title_to_actress = {}
    title_to_producer = {}
    title_to_composer = {}
    title_to_cinematographer = {}
    title_to_editor = {}
    title_to_prod_designer = {}

    for i, row in principals.iterrows():
        cat = row["category"]
        title = row["tconst"]
        nconst = row["nconst"]

        if cat == "actor":
            title_to_actor = add(title_to_actor, title, nconst)
        elif cat == "actress":
            title_to_actress = add(title_to_actress, title, nconst)
        elif cat == "producer":
            title_to_producer = add(title_to_producer, title, nconst)
        elif cat == "composer":
            title_to_composer = add(title_to_composer, title, nconst)
        elif cat == "cinematographer":
            title_to_cinematographer = add(title_to_cinematographer, title, nconst)
        elif cat == "editor":
            title_to_editor = add(title_to_editor, title, nconst)
        elif cat == "production_designer":
            title_to_prod_designer = add(title_to_prod_designer, title, nconst)

    return title_to_actor, title_to_actress, title_to_producer, title_to_composer, title_to_cinematographer, title_to_editor, title_to_prod_designer


if __name__ == "__main__":
    base_path = "./"
    df = pickle.load(open(base_path + "df_summary_top6_phrase_removed_stopwords.pkl", "rb"))

    principals = pd.read_csv(base_path + "title.principals.tsv", sep='\t')
    crew = pd.read_csv(base_path + "title.crew.tsv", sep='\t')
    cats = ["actor", "actress", "producer", "writer", "director", "composer", "cinematographer", "editor",
            "production_designer"]

    principals = principals[principals.tconst.isin(df["tconst"])]
    principals = principals[principals.category.isin(cats)]
    principals = principals.reset_index(drop=True)

    crew = crew[crew.tconst.isin(df["tconst"])]
    crew = crew.reset_index(drop=True)

    title_to_writer, title_to_director = get_from_crew(crew)
    title_to_actor, title_to_actress, title_to_producer, title_to_composer, title_to_cinematographer, title_to_editor, title_to_prod_designer = get_from_principals(
        principals)

    actor = []
    actress = []
    producer = []
    writer = []
    director = []
    composer = []
    cinematographer = []
    editor = []
    prod_designer = []

    for i, row in df.iterrows():
        title = row["tconst"]
        actor = update(actor, title, title_to_actor)
        actress = update(actress, title, title_to_actress)
        producer = update(producer, title, title_to_producer)
        writer = update(writer, title, title_to_writer)
        director = update(director, title, title_to_director)
        composer = update(composer, title, title_to_composer)
        cinematographer = update(cinematographer, title, title_to_cinematographer)
        editor = update(editor, title, title_to_editor)
        prod_designer = update(prod_designer, title, title_to_prod_designer)

    df["actor"] = actor
    df["actress"] = actress
    df["producer"] = producer
    df["writer"] = writer
    df["director"] = director
    df["composer"] = composer
    df["cinematographer"] = cinematographer
    df["editor"] = editor
    df["prod_designer"] = prod_designer

    pickle.dump(df, open(base_path + "df_summary_top6_phrase_removed_stopwords_metadata.pkl", "wb"))
    pass
