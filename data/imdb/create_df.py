import pandas as pd
import matplotlib.pyplot as plt
import pickle


def year_filter(row):
    year = row["startYear"]
    if year is None:
        return False
    try:
        if int(year) >= 1990:
            return True
        else:
            return False
    except:
        return False


def genre_filter(row):
    genres = row["genres"]
    if genres is None:
        return False
    if genres == "\\N":
        return False
    if len(genres.strip().split(",")) > 1:
        return False
    return True


if __name__ == "__main__":
    base_path = "./"
    basics = pd.read_csv(base_path + "title.basics.tsv", sep='\t')
    akas = pd.read_csv(base_path + "title.akas.tsv", sep='\t')
    ratings = pd.read_csv(base_path + "title.ratings.tsv", sep='\t')

    basics_movie = basics[basics["titleType"] == "movie"]
    basics_movie_genre = basics_movie[basics_movie.apply(genre_filter, axis=1)]

    temp_akas = akas[akas.titleId.isin(basics_movie_genre["tconst"])]
    akas_US = temp_akas[temp_akas.region == "US"]
    akas_US = akas_US.reset_index(drop=True)

    basics_movie_genre_US = basics_movie_genre[basics_movie_genre.tconst.isin(akas_US["titleId"])]

    temp_ratings = ratings[ratings.tconst.isin(basics_movie_genre_US["tconst"])]
    ratings_thresh = temp_ratings[temp_ratings.numVotes >= 100]
    ratings_thresh = ratings_thresh.reset_index(drop=True)

    basics_movie_final = basics_movie_genre_US[basics_movie_genre_US.tconst.isin(ratings_thresh["tconst"])]
    basics_movie_final = basics_movie_final.reset_index(drop=True)

    akas_US = akas_US[akas_US.titleId.isin(ratings_thresh["tconst"])]
    akas_US = akas_US.reset_index(drop=True)

    title_to_ratings_row_map = {}
    for i, row in ratings_thresh.iterrows():
        title_to_ratings_row_map[row["tconst"]] = row

    title_to_akatitle_map = {}
    for i, row in akas_US.iterrows():
        title_to_akatitle_map[row["titleId"]] = row["title"]

    aka_titles = []
    num_votes = []
    avg_ratings = []

    for i, row in basics_movie_final.iterrows():
        title = row["tconst"]
        aka_titles.append(title_to_akatitle_map[title])
        votes = title_to_ratings_row_map[title]["numVotes"]
        rating = title_to_ratings_row_map[title]["averageRating"]
        num_votes.append(votes)
        avg_ratings.append(rating)

    basics_movie_final["averageRating"] = avg_ratings
    basics_movie_final["numVotes"] = num_votes
    basics_movie_final["aka_title"] = aka_titles

    print("Length: ", len(basics_movie_final))
    print(basics_movie_final["genres"].value_counts())

    basics_movie_final["label"] = basics_movie_final["genres"]

    pickle.dump(basics_movie_final, open(base_path + "df.pkl", "wb"))
    # ratings = ratings[ratings.numVotes >= 1000]
    # ratings = ratings.reset_index(drop=True)
    #
    # temp_akas = akas[akas.titleId.isin(ratings["tconst"])]
    # akas_US = temp_akas[temp_akas.region == "US"]
    # akas_US = akas_US.reset_index(drop=True)
    #
    # temp_basics = basics[basics.tconst.isin(akas_US["titleId"])]
    # temp_basics_movie = temp_basics[temp_basics["titleType"] == "movie"]
    # basics_movie_year = temp_basics_movie[temp_basics_movie.apply(year_filter, axis=1)]
    # basics_movie_year = basics_movie_year.reset_index(drop=True)
    #
    # basics_movie_year_genre = basics_movie_year[basics_movie_year.apply(genre_filter, axis=1)]
    # basics_movie_year_genre = basics_movie_year_genre.reset_index(drop=True)
    pass
