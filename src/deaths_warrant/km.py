import pandas as pd

def event_table(times, events, entry=None):
    rows = []
    if entry is None:
        entry = [0] * len(times)
    if not (len(times) == len(events) == len(entry)):
        raise ValueError("Lengths of times, events and entry must be the same")
    for entry_time, exit_time in zip(entry, times):
        if entry_time >= exit_time:
            raise ValueError("Entry time cannot be greater than or equal to exit time")
    for t in sorted(set(times)):
        at_risk = 0
        deaths = 0
        for duration, event, entry_time in zip(times, events, entry):
            if duration >= t and entry_time < t:
                at_risk += 1
            if duration == t and event == 1:
                deaths += 1
        rows.append({"time": t, "at_risk": at_risk, "deaths": deaths})
    return pd.DataFrame(rows)

def kaplan_meier(times, events, entry=None):
    table = event_table(times, events, entry)
    survival = 1.0
    rows = []
    for row in table.itertuples(index=False):
        survival = survival * (1 - row.deaths/row.at_risk)
        rows.append({"time": row.time, "survival": survival})
    return pd.DataFrame(rows)