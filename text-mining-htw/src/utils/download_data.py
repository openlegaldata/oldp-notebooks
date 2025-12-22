import os
import csv
from datasets import load_dataset
from tqdm import tqdm
import random


def download_and_save(dataset_name="harvard-lil/cold-cases", split="train"):
    """
    Download data from HuggingFace Datasets, filter for years 2020-2024,
    and save up to the first 100,000 randomly selected rows of the filtered data to data/raw/
    with a progress bar.
    """
    # Define the path for the raw data directory
    # This assumes a directory st ructure like: project/src/scripts/download_filtered_data.py
    # and project/data/raw/
    try:
        raw_data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
        os.makedirs(raw_data_dir, exist_ok=True)
    except (TypeError, FileNotFoundError):
        # Handle cases where __file__ is not defined (e.g., in an interactive notebook)
        raw_data_dir = "data/raw"
        print(f"Could not determine script path. Using fallback directory: '{raw_data_dir}'")
        os.makedirs(raw_data_dir, exist_ok=True)

    print(f"Downloading data from: {dataset_name} ...")
    # Load the entire dataset split first in order to filter it by date
    ds = load_dataset(dataset_name, split=split)

    print("Filtering data for years...")

    # Define a function to filter records based on the date field
    def filter_by_year(example):
        target_years = {2022, 2023, 2024}  # Use integers for datetime comparison

        # The correct field name is 'date_filed' and it contains datetime.date objects
        date_value = example.get('date_filed')

        # Check if the date value exists
        if date_value is None:
            return False

        # Handle datetime.date objects (which is what the dataset actually contains)
        if hasattr(date_value, 'year'):
            return date_value.year in target_years

        # Fallback: try to handle string format if needed
        if isinstance(date_value, str) and date_value:
            try:
                year_str = date_value.split('-')[0]
                if len(year_str) == 4 and year_str.isdigit():
                    return int(year_str) in target_years
            except (IndexError, ValueError):
                pass

        return False

    # Apply the filter to the dataset
    filtered_ds = ds.filter(filter_by_year)
    print(f"Found {len(filtered_ds):,} records with dates between 2021 and 2024.")

    # Limit to 70,000 rows, but select them randomly
    num_rows_to_save = 30000
    if len(filtered_ds) > num_rows_to_save:
        # Generate random indices for sampling
        total_rows = len(filtered_ds)
        random_indices = random.sample(range(total_rows), num_rows_to_save)
        random_indices.sort()  # Sort for more efficient access

        final_ds = filtered_ds.select(random_indices)
        print(f"Randomly selected {num_rows_to_save:,} rows from {total_rows:,} filtered records.")
    else:
        final_ds = filtered_ds
        print(
            f"The number of filtered rows ({len(filtered_ds):,}) is less than {num_rows_to_save:,}, saving all of them.")

    # Define the output filename to reflect the applied filters
    output_filename = f"{dataset_name.replace('/', '_')}_{split}_2021-2024_random100k.csv"
    output_path = os.path.join(raw_data_dir, output_filename)

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Get headers from the final dataset and write them to the CSV
            headers = final_ds.column_names
            writer.writerow(headers)

            # Iterate over the final dataset with a progress bar and write rows
            for row in tqdm(final_ds, desc=f"Saving to {output_filename}"):
                writer.writerow([row[header] for header in headers])

        print(f"\nData saved successfully to: {output_path}")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    download_and_save()