import re
import csv

# Input and output file paths
input_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/fin_image_text.csv'  # Your input CSV file containing the extracted text
output_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/next_image_text.csv'  # The output CSV file

# Read the input CSV file
with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
    banned = ["I'm unable to extract", "Sure!"]
    reader = csv.reader(infile)
    headers = next(reader)  # Read the header row
    texts = [row[0] for row in reader if banned[0] not in row[0] and banned[1] not in row[0]]  # Assuming text is in the first column


# Save the cleaned texts to a new CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Extracted Text"])  # Write the header
    for text in texts:
        writer.writerow([text])

print(f"Saved cleaned text to '{output_csv}'.")