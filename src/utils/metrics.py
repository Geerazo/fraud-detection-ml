import json
from pathlib import Path

def dump_metrics(d: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
