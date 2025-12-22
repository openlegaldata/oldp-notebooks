import pandas as pd
import json
import ast
import re
from typing import Any, Union, List, Dict
import os

def safe_parse_json_string(text: str) -> Union[List, Dict, None]:
    """
    Safely parse a JSON string with multiple fallback methods.

    Args:
        text (str): The text to parse

    Returns:
        Union[List, Dict, None]: Parsed JSON object or None if parsing fails
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Clean the text
    text = text.strip()

    # Method 1: Direct JSON parsing
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 2: Handle single quotes (convert to double quotes)
    try:
        # Replace single quotes with double quotes, being careful about apostrophes
        text_fixed = re.sub(r"(?<!\\)'", '"', text)
        return json.loads(text_fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 3: Use ast.literal_eval for Python-style dictionaries
    try:
        result = ast.literal_eval(text)
        if isinstance(result, (list, dict)):
            return result
    except (ValueError, SyntaxError):
        pass

    # Method 4: Handle malformed JSON with regex fixes
    try:
        # Fix common JSON issues
        text_fixed = text
        # Fix trailing commas
        text_fixed = re.sub(r',(\s*[}\]])', r'\1', text_fixed)
        # Fix unquoted keys
        text_fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text_fixed)
        return json.loads(text_fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Method 5: Extract JSON-like patterns with regex
    try:
        # Look for array-like patterns
        array_pattern = r'\[.*\]'
        object_pattern = r'\{.*\}'

        if re.match(array_pattern, text, re.DOTALL):
            # Try to extract and parse array
            match = re.search(array_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group())
        elif re.match(object_pattern, text, re.DOTALL):
            # Try to extract and parse object
            match = re.search(object_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        pass

    print(f"Warning: Could not parse JSON string: {text[:100]}...")
    return None


def process_dataframe_with_json_column(df: pd.DataFrame, column_name: str = 'opinions') -> pd.DataFrame:
    """
    Parses a column containing a list of JSON objects (or strings of them),
    explodes the list into separate rows, and expands the JSON objects
    into new columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to process.

    Returns:
        pd.DataFrame: The processed DataFrame with new columns.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    print(f"Processing column: {column_name}")
    print(f"Original shape: {df_copy.shape}")

    # --- Step 1: Handle different data types and parse JSON strings ---
    def parse_cell_value(cell_value: Any) -> Union[List, None]:
        """Parse individual cell values into lists of dictionaries."""
        if pd.isna(cell_value):
            return None

        # If it's already a list, check if it contains dicts
        if isinstance(cell_value, list):
            # Filter out non-dict items
            dict_items = [item for item in cell_value if isinstance(item, dict)]
            return dict_items if dict_items else None

        # If it's a dict, wrap it in a list
        if isinstance(cell_value, dict):
            return [cell_value]

        # If it's a string, try to parse it
        if isinstance(cell_value, str):
            parsed = safe_parse_json_string(cell_value)
            if parsed is None:
                return None
            # Ensure we return a list
            if isinstance(parsed, dict):
                return [parsed]
            elif isinstance(parsed, list):
                # Filter to only include dict items
                dict_items = [item for item in parsed if isinstance(item, dict)]
                return dict_items if dict_items else None

        print(f"Warning: Unexpected data type {type(cell_value)} for value: {cell_value}")
        return None

    # Apply parsing to the entire column
    df_copy[column_name] = df_copy[column_name].apply(parse_cell_value)

    # Count successful parses
    successful_parses = df_copy[column_name].notna().sum()
    print(f"Successfully parsed {successful_parses} out of {len(df_copy)} rows")

    # --- Step 2: Explode the list into multiple rows ---
    # Remove rows with None/empty lists before exploding
    df_filtered = df_copy[df_copy[column_name].notna()].copy()

    if df_filtered.empty:
        print("Warning: No valid JSON data found after parsing")
        return df_filtered

    # Explode the lists into separate rows
    df_exploded = df_filtered.explode(column_name)

    # Filter out any remaining null or non-dict values after exploding
    df_exploded = df_exploded[
        df_exploded[column_name].notna() &
        df_exploded[column_name].apply(lambda x: isinstance(x, dict))
        ]

    if df_exploded.empty:
        print("Warning: No valid dictionary data found after exploding")
        return df_exploded

    print(f"Shape after exploding: {df_exploded.shape}")

    # --- Step 3: Expand the dictionaries into columns ---
    try:
        # Use pd.json_normalize for efficient and safe dictionary expansion
        json_cols = pd.json_normalize(df_exploded[column_name].tolist())

        # Align the index of the new json columns with the exploded dataframe
        json_cols.index = df_exploded.index

        print(f"Created {len(json_cols.columns)} new columns from JSON data")
        print(f"New columns: {list(json_cols.columns)}")

    except Exception as e:
        print(f"Error normalizing JSON data: {e}")
        return df_exploded

    # --- Step 4: Combine the new columns with the original data ---
    try:
        # Join the normalized JSON columns back to the exploded DataFrame
        # Drop the original column as it's now redundant
        df_final = pd.concat([
            df_exploded.drop(columns=[column_name]),
            json_cols
        ], axis=1)

        print(f"Final shape: {df_final.shape}")
        return df_final

    except Exception as e:
        print(f"Error combining columns: {e}")
        return df_exploded


def main():
    """
    Main function to create a sample DataFrame, process it, and print the result.
    """
    file_path = '../../data/processed/cleaned/legal_cases_types.csv'

    try:
        df = pd.read_csv(file_path, low_memory=False)
        print("--- Original DataFrame Info ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Show sample of the opinions column
        if 'opinions' in df.columns:
            print("\n--- Sample of 'opinions' column ---")
            sample_opinions = df['opinions'].dropna().head(3)
            for i, opinion in enumerate(sample_opinions):
                print(f"Row {i}: {str(opinion)[:200]}...")

        # Process the DataFrame
        processed_df = process_dataframe_with_json_column(df, column_name='opinions')
        cleaned_file_path = os.path.join('../../data/processed/cleaned/parse_legal_cases.csv')
        processed_df.to_csv(cleaned_file_path, index=False)
        print(f"\nParse data saved to: {cleaned_file_path}")

        print("\n--- Processed DataFrame Info ---")
        print(f"Shape: {processed_df.shape}")
        print(f"Columns: {list(processed_df.columns)}")

    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    main()