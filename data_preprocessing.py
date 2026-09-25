"""
data_preprocessing.py
----------------------
STEP 1 of the Carbon AI project.

WHAT THIS FILE DOES:
The raw dataset (SI_Data.xlsx) was compiled from ~140 different research papers
(see SI_Data_References.xlsx). Because it was copy-pasted from many PDF tables,
several numeric columns contain formatting errors, for example:
    "48.9\n0"   -> a stray newline glued two numbers together
    "31. 99"    -> a stray space inside a decimal number
    "53..5"     -> a double-dot typo
    "0.0.5"     -> two decimal points in one number

This script cleans all of that up and produces a clean CSV file that the
training script (train_model.py) can use safely.

WHY A SEPARATE FILE:
In real ML projects, data cleaning is always kept separate from model training.
This makes the project modular: if the raw data changes, you only touch this
file. If you want to change the model, you only touch train_model.py.
"""

import pandas as pd      # pandas: the standard Python library for working with tables (DataFrames)
import numpy as np        # numpy: used here for NaN (Not-a-Number, i.e. "missing value") handling

# ---------------------------------------------------------------------------
# 1. LOAD THE RAW EXCEL FILE
# ---------------------------------------------------------------------------
# pd.read_excel() reads an Excel file and turns it into a DataFrame
# (think of a DataFrame as a spreadsheet living inside Python: rows + named columns).
RAW_PATH = "data/SI_Data.xlsx"
df = pd.read_excel(RAW_PATH)

print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")


# ---------------------------------------------------------------------------
# 2. DEFINE A FUNCTION TO CLEAN A SINGLE MESSY VALUE
# ---------------------------------------------------------------------------
def clean_numeric(val):
    """
    Takes ONE cell value (which might be a clean number, or a messy string)
    and returns a clean float, or np.nan if it truly cannot be understood.
    """

    # Case 1: the cell is already empty/missing -> keep it missing.
    if pd.isna(val):
        return np.nan

    # Case 2: the cell is already a proper number (int or float) -> nothing to do.
    if isinstance(val, (int, float)):
        return float(val)

    # Case 3: the cell is text that needs cleaning.
    s = str(val).strip()                  # turn into a string and remove leading/trailing spaces

    s = s.split("\n")[0]                  # if there's a newline (two values glued together),
                                           # keep only the FIRST value, e.g. "48.9\n0" -> "48.9"

    s = s.replace(" ", "")                # remove ALL internal spaces, e.g. "31. 99" -> "31.99"

    parts = s.split(".")                  # split on the decimal point
    if len(parts) > 2:                    # more than one "." means a typo, e.g. "53..5" -> ["53","","5"]
        s = parts[0] + "." + "".join(parts[1:])   # keep only the first dot: "53" + "." + "5" = "53.5"

    try:
        return float(s)                   # finally, try converting the cleaned string to a real number
    except ValueError:
        return np.nan                     # if it still fails, mark it as missing rather than crash


# ---------------------------------------------------------------------------
# 3. APPLY THE CLEANING FUNCTION TO EVERY COLUMN THAT WAS FOUND TO BE MESSY
# ---------------------------------------------------------------------------
# These 5 columns loaded as "object" (text) dtype instead of numeric, which is
# the tell-tale sign that some rows contain non-numeric junk.
messy_columns = [
    "Oxygen",
    "Gas Flow Rate (L/min)",
    "Heating Rate (C/min)",
    "Biochar Yield (%)",
    "Bio Oil Yield (%)",
]

for col in messy_columns:
    # .apply() runs clean_numeric() on every single value in that column
    df[col] = df[col].apply(clean_numeric)

print("Cleaned messy numeric columns:", messy_columns)


# ---------------------------------------------------------------------------
# 3b. TIDY UP THE FEEDSTOCK NAME COLUMN
# ---------------------------------------------------------------------------
# Some rows have trailing spaces, e.g. "Bamboo " vs "Bamboo", which would
# otherwise be treated as two DIFFERENT feedstocks by the model. .str.strip()
# removes leading/trailing whitespace from every value in the column.
df["Feed Stock Type"] = df["Feed Stock Type"].str.strip()


# ---------------------------------------------------------------------------
# 4. DROP ROWS WHERE THE TARGET (Biochar Yield) IS MISSING
# ---------------------------------------------------------------------------
# We are building a model to PREDICT "Biochar Yield (%)". If a row has no
# yield value, it's useless for training (we'd have no correct answer to
# learn from), so we remove those rows.
before = len(df)
df = df.dropna(subset=["Biochar Yield (%)"]).copy()
print(f"Dropped {before - len(df)} rows with missing target -> {len(df)} rows remain")


# ---------------------------------------------------------------------------
# 5. DROP COLUMNS WE DON'T NEED FOR MODELING
# ---------------------------------------------------------------------------
# "References 1" / "References 2" are just numbers pointing to which paper
# (row in SI_Data_References.xlsx) the data came from. They carry no
# physical/chemical information about the pyrolysis process, so a model
# should NOT use them as predictive features.
df = df.drop(columns=["References 1", "References 2"], errors="ignore")


# ---------------------------------------------------------------------------
# 6. SAVE THE CLEAN DATA
# ---------------------------------------------------------------------------
OUT_PATH = "data/clean_biochar_data.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Saved cleaned dataset to {OUT_PATH}")
print(df.describe(include="all").T)
