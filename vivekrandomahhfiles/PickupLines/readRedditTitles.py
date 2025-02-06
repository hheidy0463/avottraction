import praw
import csv
import os
import time

# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="pZE61MfhwKiN4czLV3fpUw",  # Replace with your client ID
    client_secret="JBxK-Q6YNPoC-1r7bIbdoZ1AFnmqVQ",  # Replace with your client secret
    user_agent="flirting_data"  # Replace with your user agent
)
def get_lastid(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            lastid = row[0]

def save_lastid(lastid, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([lastid])
        
# Function to fetch titles from a subreddit
def fetch_titles(subreddit_name, limit=10, startid=None):
    subreddit = reddit.subreddit(subreddit_name)
    titles = []
    count = 0
    lastid = None
    
    # Fetch hot posts
    for post in subreddit.hot(limit=limit):
        while startid and post.id != startid:
            continue
        
        titles.append(post.title)  # Collect post titles
        count += 1
        lastid = post.id
        
        if count == 100:
            time.sleep(60)
            count = 0
            
        print(post.title)
    
    save_lastid(lastid, "C:/Users/saivi/OneDrive/Desktop/Avottraction/PickupLines/lastid.csv")
    
    return titles

# Function to save titles to a CSV file
def save_to_csv(titles, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header only if the file is being created
        if not file_exists:
            writer.writerow(["Title"])  # Write the header row

        # Write each title in a new row
        for title in titles:
            writer.writerow([title])

# Main Execution
if __name__ == "__main__":
    subreddit_name = "pickuplines"  # Subreddit name
    limit = 2000  # Number of posts to fetch
    filename = "C:/Users/saivi/OneDrive/Desktop/Avottraction/PickupLines/subreddit_titles.csv"
    try:
        start = get_lastid("C:/Users/saivi/OneDrive/Desktop/Avottraction/PickupLines/lastid.csv")
    except:
        start = None
    
    # Fetch titles from hot posts
    titles = fetch_titles(subreddit_name, limit, start)

    # Save titles to CSV, appending to the file if it exists
    save_to_csv(titles, filename)

    print(f"Titles appended to {filename}")


