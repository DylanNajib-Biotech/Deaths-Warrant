from deaths_warrant.schema import validate; import pandas as pd; import pytest

def valid_df():
    """a small table that follows every rule."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "entry_age": [40.0, 55.0, 70.0],
        "exit_age": [45.0, 60.0, 72.0],
        "status": [0, 1, 1],
        "cause": [None, "heart", "cancer"],
    })

def test_valid_table_passes():
    validate(valid_df())

def test_missing_column_fails():
    df = valid_df().drop(columns=["cause"])
    with pytest.raises(ValueError):
        validate(df)

def test_duplicate_id_fails():
    df = valid_df()
    df.loc[1, "id"] = 1
    with pytest.raises(ValueError):
        validate(df)

def test_negative_entry_age_fails():
    df = valid_df()
    df.loc[0, "entry_age"] = -5
    with pytest.raises(ValueError):
        validate(df)

def test_exit_before_entry_fails():
    df = valid_df()
    df.loc[0, "exit_age"] = 30
    with pytest.raises(ValueError):
        validate(df)

def test_bad_status_fails():
    df = valid_df()
    df.loc[0, "status"] = 2
    with pytest.raises(ValueError):
        validate(df)

def test_dead_without_cause_fails():
    df = valid_df()
    df.loc[1, "cause"] = None
    with pytest.raises(ValueError):
        validate(df)

def test_alive_with_cause_fails():
    df = valid_df()
    df.loc[0, "cause"] = "heart"
    with pytest.raises(ValueError):
        validate(df)