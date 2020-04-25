from bs4 import BeautifulSoup
import requests
import random
import pickle


def get_imd_movies(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    movies = soup.find_all("td", class_="titleColumn")
    random.shuffle(movies)
    return movies


def get_imd_summary(url, except_url):
    try:
        movie_page = requests.get(url)
        soup = BeautifulSoup(movie_page.text, 'html.parser')
        return soup.find("ul", id="plot-summaries-content").find("li").find("p").contents[0].strip()
    except:
        print("Getting summary from except_url: ", url)
        movie_page = requests.get(except_url)
        soup = BeautifulSoup(movie_page.text, 'html.parser')
        return soup.find("div", class_="summary_text").contents[0].strip()


def get_imd_movie_info(movie):
    movie_title = movie.a.contents[0]
    movie_year = movie.span.contents[0]
    movie_url = 'http://www.imdb.com' + movie.a['href']
    return movie_title, movie_year, movie_url


def imd_movie_picker():
    ctr = 0
    print("--------------------------------------------")
    for movie in get_imd_movies('http://www.imdb.com/chart/top'):
        movie_title, movie_year, movie_url = get_imd_movie_info(movie)
        movie_summary = get_imd_summary(movie_url, "")
        print(movie_title, movie_year)
        print(movie_summary)
        print("--------------------------------------------")
        ctr = ctr + 1
        if (ctr == 10):
            break


if __name__ == '__main__':
    base_path = "/data4/dheeraj/metaguide/imdb/"
    df = pickle.load(open(base_path + "df.pkl", "rb"))
    base_url = 'http://www.imdb.com/title/'
    summaries = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print("Finished: ", i)
        titleId = row["tconst"]
        movie_url = base_url + str(titleId) + "/plotsummary?ref_=tt_ov_pl#summaries"
        except_url = base_url + str(titleId) + "/"
        movie_summary = get_imd_summary(movie_url, except_url).lower()
        if len(movie_summary) == 0:
            print("Null summary: ", i, movie_url)
        summaries.append(movie_summary)

    df["summary"] = summaries
    pickle.dump(df, open(base_path + "df_summary.pkl", "wb"))
    # imd_movie_picker()
