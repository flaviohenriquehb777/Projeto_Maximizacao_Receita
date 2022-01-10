import datetime

_state = {
    "i": 0,
    "start": int(datetime.datetime(2022, 1, 5, 12, 0, 0).timestamp()),
    "end": int(datetime.datetime(2022, 6, 15, 12, 0, 0).timestamp()),
    "count": None,
}

def _ensure_count():
    if _state["count"] is None:
        try:
            with open("commit_count.txt", "r", encoding="utf-8") as f:
                _state["count"] = int(f.read().strip())
        except Exception:
            _state["count"] = 1

def callback(commit):
    _ensure_count()
    count = max(1, _state["count"])
    i = _state["i"]
    # Linear spacing between start and end across commits
    span = _state["end"] - _state["start"]
    step = 0 if count <= 1 else span / (count - 1)
    ts = int(_state["start"] + step * i)
    commit.author_date = ts
    commit.committer_date = ts
    _state["i"] = i + 1