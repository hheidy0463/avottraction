import csv

# Paths to the input CSV files
csv_file1 = 'C:/Users/saivi/OneDrive/Desktop/Avottraction/TinderLines/split-bullets.csv'
csv_file2 = "C:/Users/saivi/OneDrive/Desktop/Avottraction/PickupLines/subreddit_titles.csv"
output_csv = "C:/Users/saivi/OneDrive/Desktop/Avottraction/combined.csv"

# Open the output file for writing
with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    
    # Open the first file and write its content
    
    with open(csv_file1, mode='r', encoding='utf-8') as infile1:
        reader1 = csv.reader(infile1)
        writer.writerows(reader1)  # Write all rows including the header
    
    # Open the second file and append its content (skip the header)
    with open(csv_file2, mode='r', encoding='utf-8') as infile2:
        reader2 = csv.reader(infile2)
        next(reader2, None)  # Skip the header row
        writer.writerows(reader2)  # Write remaining rows

print(f"Combined CSV saved as '{output_csv}'.")
