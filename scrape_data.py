import arxivscraper
import pandas as pd
import pickle

if __name__ == "__main__":
    data_path = "./data/"
    scraper = arxivscraper.Scraper(category='cs', date_from='2010-06-06', date_until='2019-06-07')
    output = scraper.scrape()
    filtered_output = []
    for o in output:
        if len(o["categories"].strip().split()) > 1:
            continue
        filtered_output.append(o)
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(filtered_output, columns=cols)
    with open(data_path + "df_cs.pkl", "wb") as f:
        pickle.dump(df, f)

    df.to_csv(data_path + "cs.csv")
    pass
