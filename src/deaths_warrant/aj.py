from deaths_warrant.km import event_table; import pandas as pd

def aalen_johansen(times, causes):
    events = []
    for cause in causes:
        if cause == 0:
            events.append(0)
        else:
            events.append(1)
    table = event_table(times, events)
    cause_codes = []
    for cause in sorted(set(causes)):
        if cause != 0:
            cause_codes.append(cause)
    cif = {}
    for cause in cause_codes:
        cif[cause] = 0.0
    survival = 1.0
    rows = []
    for row in table.itertuples(index=False):
        for cause in cause_codes:
            deaths_from_cause = 0
            for duration, observed_cause in zip(times, causes):
                if duration == row.time and observed_cause == cause:
                    deaths_from_cause += 1
            cif[cause] += survival * deaths_from_cause / row.at_risk
        survival *= 1 - row.deaths / row.at_risk
        result = {"time": row.time, "survival": survival}
        for cause in cause_codes:
            result[f"cif_{cause}"] = cif[cause]
        rows.append(result)
    return pd.DataFrame(rows)