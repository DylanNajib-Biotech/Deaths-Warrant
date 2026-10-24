import pytest; import numpy as np
from deaths_warrant.what_if import remove_cause
from deaths_warrant.km import kaplan_meier
from deaths_warrant.years_lost import integrate_step

def test_remove_cause_example():
    result = remove_cause([1, 2, 2, 3, 4], [1, 2, 0, 1, 0], 1)
    table = result["curve"]
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["survival"].tolist() == pytest.approx([1, 0.75, 0.75, 0.75])
    assert result["removed_cause"] == 1
    assert "independent" in result["warning"] and "Not a causal prediction" in result["warning"]
    assert result["assumption_dependent"] is True

def test_remove_absent_cause_matches_km():
    result = remove_cause([1, 2, 2, 3, 4], [1, 2, 0, 1, 0], 99)
    table = result["curve"]
    kaplan = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert table["survival"].tolist() == pytest.approx(kaplan["survival"].tolist())

def test_remove_only_death_cause():
    result = remove_cause([1, 2, 3], [1, 1, 0], 1)
    table = result["curve"]
    assert table["survival"].tolist() == [1, 1, 1]

def test_remove_cause_all_censored():
    result = remove_cause([1, 2, 3], [0, 0, 0], 1)
    table = result["curve"]
    assert table["survival"].tolist() == [1, 1, 1]
    assert "independent" in result["warning"] and "Not a causal prediction" in result["warning"]
    assert result["assumption_dependent"] is True

def test_remove_cause_rejects_zero():
    with pytest.raises(ValueError):
        remove_cause([1, 2], [1, 0], 0)

def test_remove_cause_tied_deaths():
    result = remove_cause([1, 1, 2], [1, 2, 0], 1)
    table = result["curve"]
    assert table["time"].tolist() == [1, 2]
    assert table["survival"].tolist() == pytest.approx([2/3, 2/3])

def test_remove_cause_weighted_delayed_entry():
    result = remove_cause([2, 4, 5], [1, 2, 0], 1, [0, 0, 3], [2, 1, 3])
    table = result["curve"]
    assert table["time"].tolist() == [2, 4, 5]
    assert table["survival"].tolist() == pytest.approx([1, 0.75, 0.75])

def test_remove_cause_restricted_years_gain():
    result = remove_cause([1, 2, 2, 3, 4], [1, 2, 0, 1, 0], 1)
    table = result["curve"]
    r_curve_times = table["time"].tolist()
    r_survival_values = table["survival"].tolist()
    r_integrate = integrate_step(r_curve_times, r_survival_values, 0, 4, 1)
    kaplan = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    k_curve_times = kaplan["time"].tolist()
    k_survival_values = kaplan["survival"].tolist()
    k_integrate = integrate_step(k_curve_times, k_survival_values, 0, 4, 1)
    years_diff = r_integrate - k_integrate
    assert r_integrate == pytest.approx(3.5)
    assert k_integrate == pytest.approx(2.7)
    assert years_diff == pytest.approx(0.8)

def test_remove_cause_independent_simulation():
    rng = np.random.default_rng(42)
    heart_time = rng.exponential(scale = 5, size = 2000)
    cancer_time = rng.exponential(scale = 10, size = 2000)
    times = np.minimum(np.minimum(heart_time, cancer_time), 5)
    causes = np.where(times == 5, 0, np.where(heart_time < cancer_time, 1, 2))
    result = remove_cause(times.tolist(), causes.tolist(), 1)
    table = result["curve"]
    curve_times = table["time"].tolist()
    survival_values = table["survival"].tolist()
    integrate = integrate_step(curve_times, survival_values, 0, 5, 1)
    expected = 10 * (1 - np.exp(-5 / 10)) # Known restricted mean for the remaining cancer process
    assert integrate == pytest.approx(expected, abs = 0.2)