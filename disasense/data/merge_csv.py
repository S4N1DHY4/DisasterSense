import pandas as pd

def merge_csv_files():
    news_df = pd.read_csv("data/news_data.csv")
    news_df.drop(columns=['date','source','url'],axis=1, inplace=True)
    gdacs_df = pd.read_csv("data/gdacs_data.csv")
    gdacs_df.drop(columns=['date','url'],axis=1, inplace=True)

    print("News Data Columns:", news_df.columns.tolist())
    print("GDACS Data Columns:", gdacs_df.columns.tolist())

    news_df = news_df[~news_df['title'].str.contains(r'\[Removed\]', na=False)]
    news_df = news_df[~news_df['description'].str.contains(r'\[Removed\]', na=False)]

    gdacs_df = gdacs_df[~gdacs_df['title'].str.contains(r'\[Removed\]', na=False)]
    gdacs_df = gdacs_df[~gdacs_df['description'].str.contains(r'\[Removed\]', na=False)]


    merged_df = pd.concat([news_df, gdacs_df], ignore_index=True)

    merged_df.to_csv("data/merged_data.csv", index=False)
    print("Data has been merged and saved to merged_data.csv")


if __name__ == "__main__":
    merge_csv_files()