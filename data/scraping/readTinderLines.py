import praw
import csv
import os
import time
from openai import OpenAI

client = OpenAI()
visited = set()

# Initialize the Reddit API client
try:
    reddit = praw.Reddit(
        client_id="pZE61MfhwKiN4czLV3fpUw",  # Replace with your client ID
        client_secret="JBxK-Q6YNPoC-1r7bIbdoZ1AFnmqVQ",  # Replace with your client secret
        user_agent="flirting_data"  # Replace with your user agent
    )
except Exception as e:
    print(f"Error initializing Reddit client: {e}")
    exit(1)

# Function to fetch image URLs from a subreddit
def fetch_image_urls(subreddit_name, limit, startid=None):
    try:
        subreddit = reddit.subreddit(subreddit_name)
    except Exception as e:
        print(f"Error accessing subreddit '{subreddit_name}': {e}")
        return []
    
    image_urls = []
    lastid = None
    count = 0

    try:
        for post in subreddit.new(limit=limit):
            while startid and post.id != startid:
                continue

            if post.url.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_urls.append(post.url)
            count += 1
            lastid = post.id

            if count == 100:
                time.sleep(60)
                count = 0
            print(post.url)
    except Exception as e:
        print(f"Error fetching posts from subreddit: {e}")
    
    save_lastid(lastid, "C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/lastIdTinder.csv")
    return image_urls


def get_lastid(filename="C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/lastIdTinder.csv"):
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                return row[0]
    except Exception as e:
        print(f"Error reading last ID from file: {e}")
        return None


def save_lastid(lastid, filename):
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([lastid])
    except Exception as e:
        print(f"Error saving last ID to file: {e}")


# Function to process images and extract text using OCR
def extract_text_from_images(image_urls, csv_file="C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/image_text.csv"):
    GPTtext = ["I'm sorry, but I can't extract text from images.", "Here are the extracted messages:", "Extracted Text"]

    for url in image_urls:
        try:
            print("Extracting", url)
            responses = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "This image contains a conversation with multiple text messages. Please extract each message from the image. Do not include the sender's name, timestamp, or additional commentary."},
                            {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
                        ],
                    }
                ],
                max_tokens=300,
            )
            extracted_text = responses.choices[0].message.content
            print("Extracted text:", extracted_text)
            save_to_csv([extracted_text], csv_file)

        except Exception as e:
            print(f"Error extracting text from image {url}: {e}")


# Function to save extracted text to a CSV file
def save_to_csv(data, filename):
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Extracted Text"])

            for item in data:
                writer.writerow([item.strip()])
                print("Saved text to CSV.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    subreddit_name = "tinderpickuplines"  # Subreddit name
    total_limit = 2000  # Total number of posts to fetch
    filename = "C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/image_text.csv"

    try:
        start = get_lastid("C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/lastIdTinder.csv")
    except Exception as e:
        print(f"Error retrieving start ID: {e}")
        start = None

    try:
        image_urls = fetch_image_urls(subreddit_name, limit=total_limit, startid=start)
        print("Fetched image URLs:", image_urls)

        text = extract_text_from_images(image_urls)
        save_to_csv(text, filename)

        print(f"Processed and saved {len(image_urls)} URLs.")
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

