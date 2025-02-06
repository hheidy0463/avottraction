import csv

def remove_empty_rows(input_file, output_file):
    """Remove empty rows from a CSV file."""
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            rows = [row for row in reader if any(cell.strip() for cell in row)]  # Keep rows with non-empty cells
            
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(rows)
        
        print(f"Empty rows removed. Cleaned data saved to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = "C:/Users/saivi/OneDrive/Desktop/Avottraction/cleaned.csv"  # Replace with the path to your CSV file
output_file = "C:/Users/saivi/OneDrive/Desktop/Avottraction/c_clean.csv"  # Replace with the desired output file path
remove_empty_rows(input_file, output_file)
