from deaths_warrant.km import event_table

def test_event_table():
    table = event_table([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert table["time"].tolist() == [1, 2, 3, 4]
    assert table["at risk"].tolist() == [5, 4, 2, 1]
    assert table["deaths"].tolist() == [1, 1, 1, 0]