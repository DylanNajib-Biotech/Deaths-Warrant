from deaths_warrant.km import event_table, kaplan_meier; import pytest
from lifelines import KaplanMeierFitter

def test_event_table():
    table = event_table([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["at_risk"].tolist() == [5, 4, 2, 1]
    assert table["deaths"].tolist() == [1, 1, 1, 0]

def test_kaplan_meier():
    table = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3, 0.3])

def test_kaplan_meier_matches_lifelines():
    table = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    reference = KaplanMeierFitter().fit([1, 2, 2, 3, 4], event_observed=[1, 1, 0, 1, 0])
    expected = reference.predict(table["time"]).tolist()
    assert table["survival"].tolist() == pytest.approx(expected)

def test_kaplan_meier_all_censored():
    table = kaplan_meier([1, 2, 3], [0, 0, 0])
    assert table["survival"].tolist() == pytest.approx([1, 1, 1])

def test_kaplan_meier_tied_deaths():
    table = kaplan_meier([1, 1, 2], [1, 1, 0])
    assert table["time"].tolist() == [1, 2]
    assert table["survival"].tolist() == pytest.approx([1/3, 1/3])

def test_Kaplan_meier_unsorted():
    table = kaplan_meier([4, 2, 1, 3, 2], [0, 1, 1, 1, 0])
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3, 0.3])