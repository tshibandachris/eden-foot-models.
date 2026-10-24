import glob
import pandas as pd

print("=== DATA ENGINEERING: HISTORICAL NOMENCLATURE AUDIT ===")
found_any = False

# Scan all Bundesliga historical CSV files loaded in your repo
for path in sorted(glob.glob("**/D1.csv", recursive=True)):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        # Check both Team columns for spelling variations
        for col in ["HomeTeam", "AwayTeam"]:
            if col in df.columns:
                unique_names = df[col].dropna().unique()
                for name in unique_names:
                    if "mainz" in str(name).lower():
                        print(f"Verified in {path} ({col}): '{name}'")
                        found_any = True
    except Exception as e:
        print(f"Skipped {path} due to read exception: {e}")

if not found_any:
    print("ALERT: No historical records containing the substring 'mainz' were found.")
