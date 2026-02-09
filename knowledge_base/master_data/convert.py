print("--- 1. SCRIPT STARTED ---")

import os
import sys

# Try importing pandas with a check
try:
    import pandas as pd
    print("--- 2. PANDAS IMPORTED SUCCESSFULLY ---")
except ImportError:
    print("!!! ERROR: Pandas not installed. Run 'pip install pandas openpyxl' !!!")
    input("Press Enter to exit...")
    sys.exit()

# Configuration
INPUT_FILE = "unified_database.csv"
OUTPUT_FILE = "IRIS_Converted_Data.xlsx"

# Check if input file exists
current_dir = os.getcwd()
print(f"--- 3. CHECKING FOLDER: {current_dir} ---")
files_in_folder = os.listdir(current_dir)

if INPUT_FILE not in files_in_folder:
    print(f"!!! ERROR: Could not find '{INPUT_FILE}' in this folder.")
    print(f"    Files found: {files_in_folder}")
    print("    Please make sure the CSV file is in the same folder as this script.")
    input("Press Enter to exit...")
    sys.exit()

print(f"--- 4. READING FILE: {INPUT_FILE} ---")

try:
    # Load Data
    df = pd.read_csv(INPUT_FILE)
    print(f"    > Loaded {len(df)} rows.")

    # Clean Headers
    df.columns = [c.strip() for c in df.columns]

    # Add 'Dimension'
    if 'Dimension' not in df.columns:
        df['Dimension'] = 'Insurer'

    # Rename Columns
    print("--- 5. TRANSFORMING DATA ---")
    rename_map = {
        'Insurer': 'Entity',
        'State': 'Entity',
        'TPA': 'Entity',
        'Metric': 'Metric',
        'Value': 'Value',
        'Financial_Year': 'Financial_Year',
        'Quarter': 'Quarter',
        'Line_of_Business': 'Line_of_Business',
        'Class_of_Business': 'Class_of_Business'
    }
    df = df.rename(columns=rename_map)

    # Data Cleaning
    if 'Quarter' in df.columns:
        df['Quarter'] = df['Quarter'].astype(str).replace(['-', 'nan', 'None', ''], 'Annual')
    else:
        df['Quarter'] = 'Annual'

    # Fix Metric (Merge Unit)
    if 'Unit' in df.columns and 'Metric' in df.columns:
        df['Metric'] = df.apply(lambda x: f"{str(x['Metric']).strip()} ({str(x['Unit']).strip()})" 
                                if str(x['Unit']).strip() not in ['-', 'nan', '', 'None'] 
                                else str(x['Metric']).strip(), axis=1)

    # Fix Value (Remove commas)
    if 'Value' in df.columns:
        df['Value'] = df['Value'].astype(str).str.replace(',', '', regex=False)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Reorder
    desired_cols = ['Dimension', 'Entity', 'Financial_Year', 'Quarter', 'Metric', 'Value', 'Line_of_Business', 'Class_of_Business']
    for col in desired_cols:
        if col not in df.columns: df[col] = '-'
    final_df = df[desired_cols]

    # Save
    print(f"--- 6. SAVING TO: {OUTPUT_FILE} ---")
    final_df.to_excel(OUTPUT_FILE, index=False)
    print("!!! SUCCESS: CONVERSION COMPLETED !!!")

except Exception as e:
    print(f"!!! CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()

input("Press Enter to close...")