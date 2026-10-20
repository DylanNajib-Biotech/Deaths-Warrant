from pathlib import Path; import pytest; from deaths_warrant.nhanes import load_lmf
LMF_PATH = Path("data/raw/NHANES_2013_2014_MORT_2019_PUBLIC.dat")

# Skip, don't fail if data isn't downloaded (e.g. on another device)
needs_data = pytest.mark.skipif(not LMF_PATH.exists(), reason="LMF file not downloaded")
@needs_data
def test_row_count():
    df = load_lmf(LMF_PATH)
    assert len(df) == 10175
@needs_data
def test_alive_plus_dead_equals_eligible():
    df = load_lmf(LMF_PATH)
    alive = (df["mortstat"] == 0).sum()
    dead = (df["mortstat"] == 1).sum()
    eligible = (df["eligstat"] == 1).sum()
    assert alive + dead == eligible
@needs_data
def test_every_death_needs_a_cause():
    df = load_lmf(LMF_PATH)
    has_cause = df["ucod_leading"].notna()
    is_dead = df["mortstat"].eq(1)
    assert (has_cause == is_dead).all()
@needs_data
def test_mortstat_blank_if_not_eligible():
    df = load_lmf(LMF_PATH)
    is_blank = df["mortstat"].isna()
    not_eligible = (df["eligstat"] != 1)
    assert (is_blank == not_eligible).all()
