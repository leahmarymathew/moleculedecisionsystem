#!/usr/bin/env python3
"""
Quick inspection script to understand IQVIA workbook schema.
"""

import pandas as pd
from pathlib import Path

# Path to workbook
path = Path(__file__).resolve().parent / 'IQVIA - Leah .xlsx'

print(f"Inspecting: {path}")
print(f"Exists: {path.exists()}")

try:
    xl = pd.ExcelFile(path)
    print(f"\nSheets: {xl.sheet_names}\n")
    
    for sheet in xl.sheet_names:
        print(f"{'='*80}")
        print(f"SHEET: {sheet}")
        print(f"{'='*80}")
        
        df = pd.read_excel(path, sheet_name=sheet)
        print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nNull counts:\n{df.isnull().sum()}")
        print()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
