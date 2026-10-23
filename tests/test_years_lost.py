from deaths_warrant.years_lost import integrate_step, summarise_years; import pytest
from deaths_warrant.aj import aalen_johansen
from deaths_warrant.km import kaplan_meier
from lifelines import KaplanMeierFitter; from lifelines.utils import restricted_mean_survival_time

def test_integrate_step_survival():
    area = integrate_step([1, 2, 3, 4], [0.8, 0.6, 0.3, 0.3], 0, 4, 1)
    assert area == pytest.approx(2.7)

def test_integrate_step_partial_final_interval():
    area = integrate_step([1, 2, 3, 4], [0.8, 0.6, 0.3, 0.3], 0, 2.5, 1)
    assert area == pytest.approx(2.1)

def test_integrate_step_start_inside_interval():
    area = integrate_step([1, 2, 3, 4], [0.8, 0.6, 0.3, 0.3], 1.5, 3.5, 1)
    assert area == pytest.approx(1.15)

def test_integrate_step_before_first_update():
    area = integrate_step([1, 2, 3, 4], [0.8, 0.6, 0.3, 0.3], 0, 0.5, 1)
    assert area == pytest.approx(0.5)

def test_years_lost_by_cause():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    curve_times = table["time"].tolist()
    survival_values = table["survival"].tolist()
    years_lived = integrate_step(curve_times, survival_values, 0, 4, 1)
    cause_1 = table["cif_1"].tolist()
    years_lost_1 = integrate_step(curve_times, cause_1, 0, 4, 0)
    cause_2 = table["cif_2"].tolist()
    years_lost_2 = integrate_step(curve_times, cause_2, 0, 4, 0)
    years_summed = years_lived + years_lost_1 + years_lost_2
    assert years_lived == pytest.approx(2.7)
    assert years_lost_1 == pytest.approx(0.9)
    assert years_lost_2 == pytest.approx(0.4)
    assert years_summed == pytest.approx(4.0)

def test_integrate_step_matches_lifelines():
    table = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    curve_times = table["time"].tolist()
    survival_values = table["survival"].tolist()
    area = integrate_step(curve_times, survival_values, 0, 4, 1)
    reference = KaplanMeierFitter().fit([1, 2, 2, 3, 4], event_observed = [1, 1, 0, 1, 0])
    expected = restricted_mean_survival_time(reference, t=4)
    assert area == pytest.approx(expected)

def test_summarise_years_from_zero():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    summary = summarise_years(table, 0, 4)
    summed_years = summary["years_lived"] + summary["years_lost_by_cause"]["cif_1"] + summary["years_lost_by_cause"]["cif_2"]
    summed_window = summary["tau"] - summary["start"]
    assert summary["start"] == 0
    assert summary["tau"] == 4
    assert summary["years_lived"] == pytest.approx(2.7)
    assert summary["years_lost_by_cause"]["cif_1"] == pytest.approx(0.9)
    assert summary["years_lost_by_cause"]["cif_2"] == pytest.approx(0.4)
    assert summed_years == pytest.approx(summed_window)

def test_summarise_years_at_update():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    summary = summarise_years(table, 2, 4)
    summed_years = summary["years_lived"] + summary["years_lost_by_cause"]["cif_1"] + summary["years_lost_by_cause"]["cif_2"]
    summed_window = summary["tau"] - summary["start"]
    assert summary["start"] == 2
    assert summary["tau"] == 4
    assert summary["years_lived"] == pytest.approx(1.5)
    assert summary["years_lost_by_cause"]["cif_1"] == pytest.approx(0.5)
    assert summary["years_lost_by_cause"]["cif_2"] == pytest.approx(0.0)
    assert summed_years == pytest.approx(summed_window)

def test_summarise_years_between_updates():
    table = aalen_johansen([1, 2, 2, 3, 4], [1, 2, 0, 1, 0])
    summary = summarise_years(table, 1.5, 3.5)
    summed_years = summary["years_lived"] + summary["years_lost_by_cause"]["cif_1"] + summary["years_lost_by_cause"]["cif_2"]
    summed_window = summary["tau"] - summary["start"]
    assert summary["start"] == 1.5
    assert summary["tau"] == 3.5
    assert summary["years_lived"] == pytest.approx(1.4375)
    assert summary["years_lost_by_cause"]["cif_1"] == pytest.approx(0.1875)
    assert summary["years_lost_by_cause"]["cif_2"] == pytest.approx(0.375)
    assert summed_years == pytest.approx(summed_window)