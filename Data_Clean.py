"""
The code aims to handle systematical noise in features category, Region, Weather Condition, and Seasonality,
Q1: The relative data in those lines should be pure English, as they represent the types of products sold,
Q2: The letter 'a' is randomly subsituted by the character '@',
Q3: The letter 'e' is randomly subsituted by the character '3'.
Q4: The upper and lower cases are not consistent, and the spaces are not consistent.
Q5: Time stamp's issue
Q6: Duplicate rows are not allowed, and the key columns are Date, Store ID, Product ID, and Region.
"""

import pandas as pd
import re
from pathlib import Path

base = Path('/Users/zhangjiawei/Desktop/python_files/warwick_course')
fp_raw   = base / 'US_1_Retail.csv'
fp_final = base / 'US_1_Retail_AFTERDC.csv'
fp_conf  = base / 'US_1_Retail_REMOVED.csv'
df = pd.read_csv(fp_raw)

# Fixed Q1, 2, 3, 4
def clean_text_column(val):
    if pd.isnull(val):
        return val
    val = str(val).replace('@', 'a').replace('3', 'e')
    val = re.sub(r'[^a-zA-Z\s]', '', val) 
    val = re.sub(r'\s+', ' ', val).strip()
    return val.lower().capitalize()

for col in ['Category', 'Region', 'Weather Condition', 'Seasonality']:
    df[col] = df[col].apply(clean_text_column)

# Fixed Q5
df['Date'] = pd.to_datetime(df['Date']).dt.date

# Fixed Q6
key_cols = ['Date', 'Store ID', 'Product ID', 'Region']
dup_mask = df.duplicated(subset=key_cols, keep=False)
df_conflict = df.loc[dup_mask].copy()
df_final    = df.loc[~dup_mask].reset_index(drop=True)


df_conflict.to_csv(fp_conf, index=False)
df_final.to_csv(fp_final,  index=False)

print(f'Removed {len(df_conflict)} conflict rows; final dataset has {len(df_final)} rows.')