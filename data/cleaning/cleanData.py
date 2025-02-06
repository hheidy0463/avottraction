import praw
import csv
from openai import OpenAI
import csv
import os
import time

client = OpenAI()
src = "C:/Users/saivi/OneDrive/Desktop/Avottraction/PickupLines/.csv"

# Function to process images and extract text using OCR
def clean_text(text): #put combined scraped data here     
    responses = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Clean the following data by completing incomplete pickup lines. If the text seems to be a Header, return nothing. Remove  and replace confusing characters, keep emojis and curse words, it if appears to be neither a pickup line or Header, return as is. Do not provide any additional text. Here is the data:" + text},
                ],
            }
        ],
        max_tokens=300,
    )
    print(responses.choices[0].message.content)
    return responses.choices[0].message.content


if __name__ == "__main__":
    subreddit_name = "tinderpickuplines"  # Subreddit name
    total_limit = 2000  # Total number of posts to fetch
    batch_size = 100  # Process URLs in batches of 100
    filename = "C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/image_text.csv"
    

# Input and output file paths
input_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/combined.csv'  # Your input CSV file containing the extracted text
output_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/cleaned.csv'  # The output CSV file

# Read the input CSV file with UTF-8 encoding
with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    headers = next(reader)  # Read the header row
    texts = [row[0] for row in reader]  # Assuming text is in the first column

# Clean all extracted texts
cleaned_texts = [clean_text(text) for text in texts]

# Save the cleaned texts to a new CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Extracted Text"])  # Write the header
    for text in cleaned_texts:
        writer.writerow([text])

print(f"Saved cleaned text to '{output_csv}'.")
