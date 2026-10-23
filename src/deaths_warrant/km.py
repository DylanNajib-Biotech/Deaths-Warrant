import pandas as pd

def event_table(times, events):
    rows = []
    for t in sorted(set(times)):
        at_risk = 0
        deaths = 0
        for duration, event in zip(times, events):
            if duration >= t:
                at_risk += 1    
            if duration == t and event == 1:
                deaths += 1
        rows.append({"time": t, "at_risk": at_risk, "deaths": deaths})
    return pd.DataFrame(rows)

def kaplan_meier(times, events):
    table = event_table(times, events)
    survival = 1.0
    rows = []
    for row in table.itertuples(index=False):
        survival = survival * (1 - row.deaths/row.at_risk)
        rows.append({"time": row.time, "survival": survival})
    return pd.DataFrame(rows)