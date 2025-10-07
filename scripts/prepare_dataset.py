#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path

QC_TERMS = [
    "dépanneur",
    "poutine",
    "cégep",
    "tuque",
    "magasiner",
    "char",
    "chum",
    "blonde",
    "icitte",
    "ben là",
    "patente",
    "tabarnak",
]
MIN_LEN = 12
MAX_OUT_CHARS = 3000


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_qc(t):
    t = t.lower()
    return any(w in t for w in QC_TERMS)


def clamp(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def norm_sft(j):
    if not isinstance(j, dict) or "instruction" not in j or "output" not in j:
        return None
    instr = (j.get("instruction") or "").strip()
    inp = (j.get("input") or "").strip()
    out = j.get("output")
    if not isinstance(out, str):
        out = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    out = clamp(out, MAX_OUT_CHARS)
    if len(instr) < MIN_LEN or len(out) < MIN_LEN:
        return None
    return {"instruction": instr, "input": inp, "output": out}


def dedupe(xs, key=lambda x: x):
    s = set()
    o = []
    for it in xs:
        k = key(it)
        if k in s:
            continue
        s.add(k)
        o.append(it)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frca", default="data/raw/repo_sft.jsonl")
    ap.add_argument("--agent", default="data/raw/agent_handoff.jsonl")
    ap.add_argument("--repo", default="data/raw/repo_sft.jsonl")
    ap.add_argument("--outdir", default="data/final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", default="frca:0.50,agent:0.40,repo:0.10")
    ap.add_argument("--val_pct", type=float, default=0.06)
    ap.add_argument("--strict_qc", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed)
    ratios = {}
    for part in a.ratio.split(","):
        k, v = part.split(":")
        ratios[k.strip()] = float(v)
    sources = {"frca": a.frca, "agent": a.agent, "repo": a.repo}
    buckets = {}
    for name, path in sources.items():
        rows = []
        for j in load_jsonl(path):
            s = norm_sft(j)
            if not s:
                continue
            if a.strict_qc and name != "agent":
                if not is_qc(s["instruction"] + " " + s["output"]):
                    continue
            rows.append(s)
        rows = dedupe(
            rows, key=lambda x: sha((x["instruction"] + "|" + x["input"]).lower())
        )
        buckets[name] = rows
        print(f"[LOAD] {name}: {len(rows)}")
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        raise SystemExit("No data")
    targets = {k: int(ratios.get(k, 0) * total) for k in buckets}
    mixed = []
    for k, arr in buckets.items():
        random.shuffle(arr)
        take = min(len(arr), max(0, targets.get(k, 0)))
        mixed.extend(arr[:take])
    if len(mixed) < total:
        pool = [x for xs in buckets.values() for x in xs]
        random.shuffle(pool)
        for x in pool:
            if len(mixed) >= total:
                break
            mixed.append(x)
    random.shuffle(mixed)
    n_val = int(len(mixed) * a.val_pct)
    val = mixed[:n_val]
    train = mixed[n_val:]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "val.jsonl").open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] train={len(train)} val={len(val)} strict_qc={a.strict_qc}")


if __name__ == "__main__":
    main()
