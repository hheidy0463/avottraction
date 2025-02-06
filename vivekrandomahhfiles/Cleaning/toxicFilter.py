import pandas as pd

# Path to your CSV file
file_path = 'C:/Users/saivi/Onedrive/Desktop/Avottraction/jigsaw-toxic-comment-classification-challenge/train_data.csv/train.csv'

# Load the CSV file
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Columns to check for 'True' or '1'
filter_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Filter rows where any of these columns have a value of 1
filtered_df = df.loc[df[filter_columns].any(axis=1)]

# Save the filtered data to a new CSV
output_path = 'C:/Users/saivi/Onedrive/Desktop/Avottraction/filtered_data.csv'
if not filtered_df.empty:
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered data saved successfully at {output_path}")
else:
    print("No rows met the filtering criteria.")

