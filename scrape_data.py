import arxivscraper
import pandas as pd
import pickle

if __name__ == "__main__":
    data_path = "/data4/dheeraj/metaguide/"
    scraper = arxivscraper.Scraper(category='cs', date_from='2014-06-06', date_until='2019-06-07', timeout=86400)
    output = scraper.scrape()
    filtered_output = []
    for o in output:
        if len(o["categories"].strip().split()) > 1:
            continue
        filtered_output.append(o)
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(filtered_output, columns=cols)
    print("Length of Dataframe", len(df))
    with open(data_path + "df_cs_2014.pkl", "wb") as f:
        pickle.dump(df, f)

    df.to_csv(data_path + "cs_2014.csv")
    pass
