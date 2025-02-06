import re
import csv

# Function to clean text by removing numbers and bullets before the first letter
def clean_text(text):
    # Removing bullets and numbers before the first letter
    cleaned = re.sub(r"^\s*[-\d]+\.*\s*", "", text)
    return cleaned

# Input and output file paths
input_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/split.csv'  # Your input CSV file containing the extracted text
output_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/split-bullets.csv'  # The output CSV file

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
