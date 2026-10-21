import pandas as pd

# Positions from NHS R_ReadInProgramAllSurveys.R, converted to 0-based, end-exclusive
LMF_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
LMF_NAMES = ["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes", "hyperten", "permth_int", "permth_exm"]

def load_lmf(path):
    """Read an NHANES public-use linked mortality dat. file."""
    return pd.read_fwf(path, colspecs=LMF_COLSPECS, names=LMF_NAMES,
        na_values=["", "."], dtype={"ucod_leading": "string"}) # Keep leading zeroes: "001", not "1"

DEMO_COLUMNS = {
    "SEQN": "seqn",
    "RIAGENDR": "sex",
    "RIDAGEYR": "age_years",
    "WTMEC2YR": "weight_exam",
    "SDMVPSU": "psu",
    "SDMVSTRA": "stratum",
}

def load_demo(path):
    """Read an NHANES demographics .xpt file, keeping only the columns we use."""
    df = pd.read_sas(path, format="xport")
    df = df[list(DEMO_COLUMNS)].rename(columns=DEMO_COLUMNS)
    df["seqn"] = df["seqn"].astype(int) # .xpt stores IDs as decimals
    return df