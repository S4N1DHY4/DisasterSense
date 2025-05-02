import requests
import pandas as pd
import datetime
import os

API_KEY = "YOUR_API_KEY"  
KEYWORDS = "disaster OR earthquake OR flood OR wildfire OR hurricane"  
URL = f"https://newsapi.org/v2/everything"

def fetch_news_data():
    params = {
        'q': KEYWORDS,
        'from': (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
        'sortBy': 'publishedAt',
        'apiKey': API_KEY
    }
    
    response = requests.get(URL, params=params)
    if response.status_code == 200:
        data = response.json()
        
        if 'articles' not in data:
            print("Unexpected response structure:", data)
            return pd.DataFrame()  

        if len(data['articles']) == 0:
            print("No articles found for the given keywords and date range.")
            return pd.DataFrame()
        
        articles = []
        for item in data['articles']:
            article = {
                'title': item['title'],
                'date': item['publishedAt'],
                'source': item['source']['name'],
                'description': item['description'],
                'url': item['url']
            }
            articles.append(article)
        return pd.DataFrame(articles)
    else:
        print(f"Failed to fetch data from Google News. Status Code: {response.status_code}")
        print("Response:", response.text)
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_news_data()
    file_path = "data/news_data.csv"
    
    if not df.empty:
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', index=False, header=False)
        print("Google News disaster data fetched and appended to news_data.csv")
    else:
        print("No data fetched or data was invalid.")
