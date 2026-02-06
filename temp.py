import pandas as pd
import random

# 1. Define the Standard Columns
columns = ["Insurer", "Financial_Year", "Quarter", "Line_of_Business", "Metric", "Value", "Unit"]

# 2. Create Dummy Data
data = [
    # Star Health Data
    ["Star Health", "FY 2024-25", "Q1", "Health", "Solvency Ratio", 2.2, "Ratio"],
    ["Star Health", "FY 2024-25", "Q1", "Health", "GDPI", 1520.5, "Cr"],
    ["Star Health", "FY 2024-25", "Q1", "Health", "Incurred Claims Ratio", 65.4, "%"],
    ["Star Health", "FY 2024-25", "Q1", "Health", "Net Profit", 120.0, "Cr"],

    # ICICI Lombard Data
    ["ICICI Lombard", "FY 2024-25", "Q1", "Motor", "Incurred Claims Ratio", 78.2, "%"],
    ["ICICI Lombard", "FY 2024-25", "Q1", "Health", "Solvency Ratio", 2.5, "Ratio"],
    ["ICICI Lombard", "FY 2024-25", "Q1", "Fire", "GDPI", 450.0, "Cr"],

    # Niva Bupa Data
    ["Niva Bupa", "FY 2023-24", "Q4", "Health", "Solvency Ratio", 1.9, "Ratio"],
    ["Niva Bupa", "FY 2023-24", "Q4", "Health", "Commission Ratio", 14.5, "%"],
    
    # Acko Data
    ["Acko", "FY 2024-25", "Q2", "Motor", "GDPI", 800.0, "Cr"],
    ["Acko", "FY 2024-25", "Q2", "Motor", "Net Profit", -45.0, "Cr"]
]

# 3. Create DataFrame
df = pd.DataFrame(data, columns=columns)

# 4. Save to Excel
filename = "sample_submission.xlsx"
df.to_excel(filename, index=False)

print(f"✅ Generated '{filename}' with {len(df)} rows.")
print("👉 Move this file to 'knowledge_base/raw_submissions/' to test.")