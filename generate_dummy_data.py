import pandas as pd
import os
import random

# 1. Setup Folder
TARGET_FOLDER = "knowledge_base/raw_submissions"
if not os.path.exists(TARGET_FOLDER):
    os.makedirs(TARGET_FOLDER)
    print(f"📁 Created folder: {TARGET_FOLDER}")

# 2. List of SAHIs (Standalone Health Insurers)
insurers = [
    "Aditya Birla Health",
    "Care Health",
    "Star Health",
    "Niva Bupa",
    "Manipal Cigna",
    "Galaxy Health",
    "Narayana Health"
]

# 3. Metrics to Simulate
metrics_config = [
    {"name": "Solvency Ratio", "unit": "Ratio", "min": 1.5, "max": 2.5, "is_int": False},
    {"name": "GDPI", "unit": "Cr", "min": 500, "max": 15000, "is_int": True},
    {"name": "Net Profit", "unit": "Cr", "min": -200, "max": 800, "is_int": True},
    {"name": "Incurred Claims Ratio", "unit": "%", "min": 60, "max": 95, "is_int": True},
    {"name": "Commission Ratio", "unit": "%", "min": 10, "max": 18, "is_int": True},
    {"name": "Combined Ratio", "unit": "%", "min": 95, "max": 115, "is_int": True}
]

years = ["FY 2024", "FY 2025"]
quarters = ["Q1", "Q2", "Q3", "Q4"]

# 4. Generator Loop
for insurer in insurers:
    data = []
    
    for year in years:
        for q in quarters:
            # Skip future dates (e.g., FY 2025 Q3/Q4)
            if year == "FY 2025" and q in ["Q3", "Q4"]: continue
            
            for m in metrics_config:
                # Generate random value based on insurer size variation
                # (Star & Care get slightly larger numbers for realism)
                multiplier = 1.0
                if insurer in ["Star Health", "Care Health"]: multiplier = 1.5
                elif insurer in ["Galaxy Health", "Narayana Health"]: multiplier = 0.3
                
                raw_val = random.uniform(m["min"], m["max"])
                
                # Logic for monetary values vs ratios
                if m["unit"] == "Cr":
                    final_val = int(raw_val * multiplier)
                else:
                    final_val = round(raw_val, 2)
                    if m["is_int"]: final_val = int(final_val)

                row = {
                    "Insurer": insurer,
                    "Financial_Year": year,
                    "Quarter": q,
                    "Line_of_Business": "Health",
                    "Metric": m["name"],
                    "Value": final_val,
                    "Unit": m["unit"]
                }
                data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    filename = f"{insurer.replace(' ', '_')}_Submission.xlsx"
    filepath = os.path.join(TARGET_FOLDER, filename)
    df.to_excel(filepath, index=False)
    
    print(f"✅ Generated: {filename} ({len(df)} rows)")

print("\n🎉 All 7 insurer files are ready in 'knowledge_base/raw_submissions'.")
print("👉 You can now go to /data and click 'Sync Data'!")