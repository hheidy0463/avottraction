import csv
import random

def shuffle(list1, list2):
    # Zip the two lists together, shuffle, and unzip back
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    
    return combined

def mix(file1, file2, output_file):
    """Combine two CSV files and stack the rows in a single column."""
    try:
        # Read the first file
        with open(file1, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            rows = [row[0] for row in reader if row]  # Take the first column
        
        # Read the second file, and select the second column (if it exists)
        with open(file2, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            rows2 = [row[1] if len(row) > 1 else '' for row in reader if row]  # Take the second column if it exists

        # Slice rows2 to only take the first 7500 elements
        rows2 = rows2[:7500]

        print(rows, rows2)  # Debugging to check the lists

        # Combine the two lists and make sure they are placed in the same column
        merged_rows = [[item] for item in rows + rows2]  # Concatenate both lists and wrap each item in a list

        # Optionally, limit the number of rows being written (for debugging purposes)
        print(merged_rows[:2])  # Print the first two merged rows to verify

        # Write the merged rows to the output file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(merged_rows)
        
        print(f"Fused data saved to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
rizz_file = "C:/Users/saivi/OneDrive/Desktop/Avottraction/cleaned.csv"  
base_file = "C:/Users/saivi/OneDrive/Desktop/Avottraction/jigsaw-toxic-comment-classification-challenge/train_data.csv/train.csv"
output_file = "C:/Users/saivi/OneDrive/Desktop/Avottraction/10000.csv"  
mix(rizz_file, base_file, output_file)
