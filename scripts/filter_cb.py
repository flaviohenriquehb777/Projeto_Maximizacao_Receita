import json

# Lazy-load mapping from commit SHA -> epoch seconds
_MAP = None

def _load_map():
    global _MAP
    if _MAP is None:
        with open("commit_dates.json", "r", encoding="utf-8") as f:
            _MAP = json.load(f)
    return _MAP

# Executed once per commit by git-filter-repo
def callback(commit):
    m = _load_map()
    oid = commit.original_id
    try:
        oid = oid.decode()
    except Exception:
        pass
    ts = m.get(oid)
    if ts is not None:
        ts_int = int(ts)
        commit.author_date = ts_int
        commit.committer_date = ts_int