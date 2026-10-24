from deaths_warrant.aj import aalen_johansen; import pytest
from lifelines import AalenJohansenFitter
from deaths_warrant.km import kaplan_meier

def test_aalen_johansen_example():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3, 0.3])
    assert table["cif_1"].tolist() == pytest.approx([0.2, 0.2, 0.5, 0.5])
    assert table["cif_2"].tolist() == pytest.approx([0.0, 0.2, 0.2, 0.2])

def test_aalen_johansen_probabilities_sum_to_one():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    total = table["survival"] + table["cif_1"] + table["cif_2"]
    assert total.tolist() == pytest.approx([1, 1, 1, 1])

def test_aalen_johansen_unsorted():
    table = aalen_johansen([4, 2, 1, 3, 2], [0, 2, 1, 1, 0])
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3, 0.3])
    assert table["cif_1"].tolist() == pytest.approx([0.2, 0.2, 0.5, 0.5])
    assert table["cif_2"].tolist() == pytest.approx([0.0, 0.2, 0.2, 0.2])

def test_aalen_johansen_matches_lifelines():
    table = aalen_johansen([1, 2, 3, 4, 5], [1, 2, 0, 1, 0])
    for cause in [1, 2]:
        reference = AalenJohansenFitter().fit(
                [1, 2, 3, 4, 5], event_observed=[1, 2, 0, 1, 0], event_of_interest = cause
            )
        expected = reference.predict(table["time"]).tolist()
        assert table[f"cif_{cause}"].tolist() == pytest.approx(expected)

def test_aalen_johansen_all_censored():
    table = aalen_johansen([1, 2, 3], [0, 0, 0])
    assert table["time"].tolist() == pytest.approx([1, 2, 3])
    assert table["survival"].tolist() == pytest.approx([1, 1, 1])

def test_aalen_johansen_tied_causes():
    table = aalen_johansen([1, 1, 2], [1, 2, 0])
    assert table["time"].tolist() == pytest.approx([1, 2])
    assert table["survival"].tolist() == pytest.approx([1/3, 1/3])
    for cause in [1, 2]:
        assert table[f"cif_{cause}"].tolist() == pytest.approx([1/3, 1/3])

def test_aalen_johansen_cifs_match_km():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    km_table = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    total = table["cif_1"] + table["cif_2"]
    km_subtraction = (1 - km_table["survival"]).tolist()
    assert total.tolist() == pytest.approx(km_subtraction)

def test_aalen_johansen_delayed_entry():
    table = aalen_johansen([2, 4, 5], [1, 2, 0], [0, 0, 3])
    assert table["time"].tolist() == [2, 4, 5]
    assert table["survival"].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert table["cif_1"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert table["cif_2"].tolist() == pytest.approx([0.0, 0.25, 0.25])

def test_aalen_johansen_weighted():
    table = aalen_johansen([1, 2, 3], [1, 2, 0], weights = [2, 1, 3])
    assert table["time"].tolist() == [1, 2, 3]
    assert table["survival"].tolist() == pytest.approx([2/3, 0.5, 0.5])
    assert table["cif_1"].tolist() == pytest.approx([1/3, 1/3, 1/3])
    assert table["cif_2"].tolist() == pytest.approx([0, 1/6, 1/6])

def test_aalen_johansen_weighted_matches_lifelines():
    table = aalen_johansen([1, 2, 3], [1, 2, 0], weights = [2, 1, 3])
    for cause in [1, 2]:
        reference = AalenJohansenFitter().fit(
            [1, 2, 3], event_observed = [1, 2, 0], event_of_interest = cause, weights = [2, 1, 3]
        )
        expected = reference.predict(table["time"].tolist())
        assert table[f"cif_{cause}"].tolist() == pytest.approx(expected)

def test_aalen_johansen_weight_scale_invariance():
    table_1 = aalen_johansen([1, 2, 3], [1, 2, 0], weights = [2, 1, 3])
    table_2 = aalen_johansen([1, 2, 3], [1, 2, 0], weights = [0.2, 0.1, 0.3])
    assert table_1["time"].tolist() == table_2["time"].tolist()
    assert table_1["survival"].tolist() == pytest.approx(table_2["survival"].tolist())
    assert table_1["cif_1"].tolist() == pytest.approx(table_2["cif_1"].tolist())
    assert table_1["cif_2"].tolist() == pytest.approx(table_2["cif_2"].tolist())

def test_aalen_johansen_weighted_delayed_entry():
    table = aalen_johansen([2, 4, 5], [1, 2, 0], [0, 0, 3], [2, 1, 3])
    assert table["time"].tolist() == [2, 4, 5]
    assert table["survival"].tolist() == pytest.approx([1/3, 1/4,1/4])
    assert table["cif_1"].tolist() == pytest.approx([2/3, 2/3, 2/3])
    assert table["cif_2"].tolist() == pytest.approx([0, 1/12, 1/12])

    