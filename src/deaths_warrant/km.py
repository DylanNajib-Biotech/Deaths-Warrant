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
        rows.append({"time": t, "at risk": at_risk, "deaths": deaths})
    return pd.DataFrame(rows)