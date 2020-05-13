from bs4 import BeautifulSoup
import requests
import random
import pickle
import itertools
import numpy


def getSoup(url):
    """
    Utility function which takes a url and returns a Soup object.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    return soup


def minMax(a):
    '''Returns the index of negative and positive review.'''

    # get the index of least rated user review

    inds = numpy.argsort(a)
    minpos = inds[-2]

    # get the index of highest rated user review
    maxpos = inds[-1]

    return minpos, maxpos


def getReviews(soup):
    '''Function returns a negative and positive review for each movie.'''

    # get a list of user ratings
    user_review_ratings = [tag.previous_element for tag in
                           soup.find_all('span', attrs={'class': 'point-scale'})]

    # find the index of negative and positive review
    n_index, p_index = minMax(list(map(int, user_review_ratings)))

    # get the review tags
    user_review_list = soup.find_all('a', attrs={'class': 'title'})

    # get the negative and positive review tags
    n_review_tag = user_review_list[n_index]
    p_review_tag = user_review_list[p_index]

    # return the negative and positive review link
    n_review_link = "https://www.imdb.com" + n_review_tag['href']
    p_review_link = "https://www.imdb.com" + p_review_tag['href']

    return p_review_link


def getReviewText(review_url):
    '''Returns the user review text given the review url.'''

    # get the review_url's soup
    soup = getSoup(review_url)

    # find div tags with class text show-more__control
    tag = soup.find('div', attrs={'class': 'text show-more__control'})

    return tag.getText()


def get_imd_review(url):
    movie_soups = getSoup(url)
    movie_review_list = getReviews(movie_soups)
    review_texts = getReviewText(movie_review_list)
    return review_texts


if __name__ == '__main__':
    base_path = "/data4/dheeraj/metaguide/imdb/"
    # base_path = "./"
    df = pickle.load(open(base_path + "df_summary_top6.pkl", "rb"))
    base_url = 'https://www.imdb.com/title/'
    reviews = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print("Finished: ", i)
        titleId = row["tconst"]
        review_url = base_url + str(titleId) + "/reviews"
        movie_review = get_imd_review(review_url).lower()
        if len(movie_review) == 0:
            print("Null Review: ", i, review_url)
        reviews.append(movie_review)

    df["review"] = reviews
    pickle.dump(df, open(base_path + "df_summary_top6_reviews.pkl", "wb"))
    # imd_movie_picker()
