# Global Disaster Alert and Coordination System (GDACS)
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import os

def fetch_gdacs_data():
    url = "https://www.gdacs.org/xml/rss.xml"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)

        events = []
        for item in root.findall(".//item"):
            title = item.find("title").text if item.find("title") is not None else "N/A"
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else "N/A"
            link = item.find("link").text if item.find("link") is not None else "N/A"
            description = item.find("description").text if item.find("description") is not None else "N/A"
            
            events.append({
                'title': title,
                'date': pub_date,
                'description': description,
                'url': link
            })
        
        return pd.DataFrame(events)
    else:
        print(f"Failed to fetch data from GDACS. Status Code: {response.status_code}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = fetch_gdacs_data()
    file_path = "data/gdacs_data.csv"
    
    if not df.empty:
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', index=False, header=False)
        print("GDACS disaster data fetched and appended to gdacs_data.csv")
    else:
        print("No data fetched or data was invalid.")
