import pandas as pd

# Positions from NHS R_ReadInProgramAllSurveys.R, converted to 0-based, end-exclusive
LMF_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
LMF_NAMES = ["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes", "hyperten", "permth_int", "permth_exm"]

def load_lmf(path):
    """Read an NHANES public-use linked mortality dat. file."""
    return pd.read_fwf(path, colspecs=LMF_COLSPECS, names=LMF_NAMES,
        na_values=["", "."], dtype={"ucod_leading": "string"}) # Keep leading zeroes: "001", not "1"
