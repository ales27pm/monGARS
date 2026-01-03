#!/usr/bin/env python3
# Lite repo analyzer: mines SFT, agent-handoff, embeddings, and writes a DOT graph.
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

OUT = Path("data/ultimate")
RAW = OUT / "raw_texts"
PROC = OUT / "processed_repo"
for d in (RAW, PROC):
    d.mkdir(parents=True, exist_ok=True)
SFT = PROC / "sft_repo.jsonl"
AGT = PROC / "agent_instruct_repo.jsonl"
EMB = PROC / "embeddings_repo.jsonl"
PROV = PROC / "provenance.csv"
DOT = OUT / "interaction_graph.dot"
PNG = OUT / "interaction_graph.png"

if os.environ.get("CONFIRM_SCAN", "") != "YES":
    print("ABORT: set CONFIRM_SCAN=YES")
    sys.exit(2)

EXTS = {
    "md",
    "rst",
    "txt",
    "json",
    "yml",
    "yaml",
    "py",
    "sh",
    "cfg",
    "ini",
    "toml",
    "sql",
    "js",
    "ts",
}


def is_text(p):
    try:
        with p.open("rb") as f:
            return b"\x00" not in f.read(4096)
    except OSError as exc:
        logger.debug(
            "ultimate_repo_analyzer.read_failed",
            extra={"path": str(p), "error": str(exc)},
        )
        return False


root = Path(".").resolve()
files = []
if (root / ".git").exists():
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True)
        for rel in out.splitlines():
            p = root / rel
            if p.suffix.lstrip(".") in EXTS and p.exists() and is_text(p):
                dst = RAW / p.relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes())
                files.append(dst)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning(
            "ultimate_repo_analyzer.git_ls_failed",
            extra={"error": str(exc)},
        )
if not files:
    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lstrip(".") in EXTS
            and is_text(p)
            and ".git" not in p.parts
        ):
            dst = RAW / p.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
            files.append(dst)

DIALOG = re.compile(
    r"^\s*(User|Utilisateur|Client|Moi|Tu|Vous|Assistant|System|Bot|Agent)\s*[:\-—]\s*(.+)",
    re.I,
)
PIPE = re.compile(
    r"(workflow|pipeline|job|stage|steps|run:|script:|entrypoint|commands?)", re.I
)
JSONL = re.compile(r'^\s*[\{\[]\s*".*')


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


sft_rows = []
ag_rows = []
emb_rows = []
prov = []

for f in files:
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning(
            "ultimate_repo_analyzer.read_text_failed",
            extra={"path": str(f), "error": str(exc)},
        )
        continue
    lines = text.splitlines()

    # dialogues → SFT
    cur = []
    for i, ln in enumerate(lines):
        if DIALOG.match(ln):
            cur.append((i + 1, ln.strip()))
        else:
            if cur:
                instr = None
                outs = []
                for _, l in cur:
                    m = DIALOG.match(l)
                    who = m.group(1).lower()
                    content = m.group(2).strip()
                    if instr is None and re.match(
                        r"(user|utilisateur|client|moi|tu|vous)", who, re.I
                    ):
                        instr = content
                    else:
                        outs.append(content)
                if instr and outs:
                    rec = {"instruction": instr, "input": "", "output": " ".join(outs)}
                    sft_rows.append(rec)
                    prov.append(
                        [
                            sha(json.dumps(rec, ensure_ascii=False)),
                            str(f.relative_to(RAW)),
                            cur[0][0],
                            cur[-1][0],
                            "sft_dialog",
                            "auto",
                        ]
                    )
                cur = []
    if cur:
        instr = None
        outs = []
        for _, l in cur:
            m = DIALOG.match(l)
            who = m.group(1).lower()
            content = m.group(2).strip()
            if instr is None and re.match(
                r"(user|utilisateur|client|moi|tu|vous)", who, re.I
            ):
                instr = content
            else:
                outs.append(content)
        if instr and outs:
            rec = {"instruction": instr, "input": "", "output": " ".join(outs)}
            sft_rows.append(rec)
            prov.append(
                [
                    sha(json.dumps(rec, ensure_ascii=False)),
                    str(f.relative_to(RAW)),
                    cur[0][0],
                    cur[-1][0],
                    "sft_dialog",
                    "auto",
                ]
            )

    # pipeline fragments → agent samples
    curp = []
    for i, ln in enumerate(lines):
        if (
            PIPE.search(ln)
            or JSONL.match(ln)
            or ln.strip().startswith(("steps:", "- run:", "script:"))
        ):
            curp.append((i + 1, ln))
        else:
            if curp:
                block = "\n".join(l for _, l in curp)
                rec = {
                    "instruction": "Interpret this pipeline fragment and return structured steps as JSON. Preserve tool names and env vars.",
                    "input": block,
                    "output": {"steps": [], "notes": "AUTO"},
                }
                ag_rows.append(rec)
                prov.append(
                    [
                        sha(json.dumps(rec, ensure_ascii=False)),
                        str(f.relative_to(RAW)),
                        curp[0][0],
                        curp[-1][0],
                        "agent_pipeline",
                        "auto",
                    ]
                )
                curp = []
    if curp:
        block = "\n".join(l for _, l in curp)
        rec = {
            "instruction": "Interpret this pipeline fragment and return structured steps as JSON. Preserve tool names and env vars.",
            "input": block,
            "output": {"steps": [], "notes": "AUTO"},
        }
        ag_rows.append(rec)
        prov.append(
            [
                sha(json.dumps(rec, ensure_ascii=False)),
                str(f.relative_to(RAW)),
                curp[0][0],
                curp[-1][0],
                "agent_pipeline",
                "auto",
            ]
        )

    # paragraphs → embeddings
    para = []
    start = 1
    for i, ln in enumerate(lines):
        if ln.strip() == "":
            if para:
                t = "\n".join(para).strip()
                if len(t) >= 40:
                    emb_rows.append({"text": t, "source": str(f.relative_to(RAW))})
                    prov.append(
                        [
                            sha(t),
                            str(f.relative_to(RAW)),
                            start,
                            i,
                            "embedding_paragraph",
                            "auto",
                        ]
                    )
                para = []
                start = i + 2
        else:
            para.append(ln)
    if para:
        t = "\n".join(para).strip()
        if len(t) >= 40:
            emb_rows.append({"text": t, "source": str(f.relative_to(RAW))})
            prov.append(
                [
                    sha(t),
                    str(f.relative_to(RAW)),
                    start,
                    len(lines),
                    "embedding_paragraph",
                    "auto",
                ]
            )


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write_jsonl(SFT, sft_rows)
write_jsonl(AGT, ag_rows)
write_jsonl(EMB, emb_rows)
with PROV.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["record_id", "source_file", "start_line", "end_line", "type", "note"])
    w.writerows(prov)

with DOT.open("w", encoding="utf-8") as f:
    f.write('digraph interactions {\n  "repo" [label="repo"];\n')
    for i in range(min(20, len(sft_rows))):
        f.write('  "repo" -> "sft_' + str(i) + '";\n')
    f.write("}\n")
try:
    subprocess.run(["dot", "-Tpng", str(DOT), "-o", str(PNG)], check=True)
except (OSError, subprocess.CalledProcessError) as exc:
    logger.warning(
        "ultimate_repo_analyzer.graphviz_failed",
        extra={"error": str(exc)},
    )

print(f"OK: SFT={len(sft_rows)} AGENT={len(ag_rows)} EMB={len(emb_rows)}")
print("OUT:", SFT, AGT, EMB, PROV, DOT, PNG if PNG.exists() else "(no PNG)")
