import tweepy
import time
import csv
import logging
from googletrans import Translator  
import sys
from http.client import RemoteDisconnected
import asyncio

bearer_token = "YOUR_BEARER_TOKEN" #Replace with your own API V2 bearer token
client = tweepy.Client(bearer_token=bearer_token)

trusted_accounts = ["Top_Disaster","ndmaindia"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
sys.stdout.reconfigure(encoding='utf-8')
translator = Translator()
async def translate_text(content):
    try:
        translated = await translator.translate(content, src='auto', dest='en')
        logging.info(f"Translated content: {translated.text}")
        return translated.text
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return content

def translate_to_english(content):
    return asyncio.run(translate_text(content))

def fetch_disaster_tweets(client, accounts):
    try:
        with open("data/merged_data.csv", mode='a', encoding="utf-8", newline='') as cs
        v_file:
            fieldnames = ['title', 'description']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)

            logging.info("Starting to fetch disaster-related tweets...")
            for account in accounts:
                user = client.get_user(username=account)
                max_tweets = 5
                tweet_count = 0
                pagination_token = None

                while tweet_count < max_tweets:
                    try:
                        response = client.get_users_tweets(
                            id=user.data.id,
                            max_results=10,
                            pagination_token=pagination_token,
                            tweet_fields=["created_at"]
                        )

                        if response.data:
                            for tweet in response.data:
                                if tweet_count >= max_tweets:
                                    break

                                content = tweet.text.lower()
                                if any(keyword in content for keyword in ["earthquake", "flood", "hurricane", "disaster", "rescue", "relief"]):
                                    content_in_english = translate_to_english(tweet.text)
                                    title = f"Tweet by @{account} on {tweet.created_at.strftime('%Y-%m-%d') if tweet.created_at else 'Unknown'}"
                                    description = f"{content_in_english}"
                                    writer.writerow({'title': title, 'description': description})
                                    tweet_count += 1
                        
                        if "next_token" in response.meta:
                            pagination_token = response.meta["next_token"]
                        else:
                            break
                        
                    except tweepy.TooManyRequests as e:
                        reset_time = int(e.response.headers.get("x-rate-limit-reset"))
                        wait_time = max(0, reset_time - int(time.time()))
                        logging.warning("Rate limit exceeded. Waiting to retry...")
                        time.sleep(wait_time)
                    except RemoteDisconnected:
                        logging.warning("Remote end closed connection. Retrying in 1 minute...")
                        time.sleep(60)
                    except Exception as e:
                        logging.error(f"An error occurred: {e}")
                        break
            logging.info("Finished fetching tweets.")
    except Exception as e:
        logging.error(f"A critical error occurred: {e}")

if __name__ == "__main__":
    logging.info("Script execution started.")
    try:
        fetch_disaster_tweets(client, trusted_accounts)

        logging.info("Script execution completed successfully.")
    except Exception as e:
        logging.error(f"Critical error in script execution: {e}")
