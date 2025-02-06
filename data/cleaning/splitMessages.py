import re
import csv

def split_lists(extracted_text):
    # Regular expressions to capture numbered or bullet-point lists
    numbered_list_pattern = r"(\d+\.\s.*?)(?=(?:\n\d+\.\s)|\Z)"
    bullet_list_pattern = r"(- .+?)(?=(?:\n- )|\Z)"

    # Combine both numbered and bullet patterns
    combined_pattern = rf"{numbered_list_pattern}|{bullet_list_pattern}"

    # Extract matches
    matches = re.findall(combined_pattern, extracted_text, flags=re.DOTALL)

    # Clean and organize the matches
    extracted_items = []
    for match in matches:
        # match[0] corresponds to numbered_list_pattern, match[1] corresponds to bullet_list_pattern
        item = match[0] if match[0] else match[1]
        extracted_items.append(item.strip())

    return extracted_items


# Input and output file paths
input_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/next_image_text.csv'  # Your input CSV file containing the extracted text
output_csv = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/split.csv'  # The output CSV file

# Read the input CSV file
texts = []
with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    headers = next(reader)  # Read the header row
    for row in reader:
        # Process each row and split lists
        if row:  # Ensure row is not empty
            for text in split_lists(row[0]):
                texts.append(text)

# Save the cleaned texts to a new CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Extracted Text"])  # Write the header
    for text in texts:
        writer.writerow([text])

print(f"Saved cleaned text to '{output_csv}'.")
