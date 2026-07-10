"""
reorganize.py
─────────────
Classifies every folder in D:/divided by name pattern and moves it into
a logical agency/type subfolder hierarchy.

Step 1 — dry run (default):  prints every planned move, nothing changes.
Step 2 — execute:            pass --execute to actually move folders.
Step 3 — undo:               pass --undo to reverse (reads the move log).

Usage
-----
    python reorganize.py                     # dry-run, print plan
    python reorganize.py --execute           # move everything
    python reorganize.py --undo              # reverse all moves (uses move_log.json)
    python reorganize.py --execute --root D:/somewhere_else
"""

import re
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ── Target root (all new subfolders are created inside here) ─────────────────
DEFAULT_ROOT = os.environ.get("UAP_PIPELINE_ROOT", ".")

# ── Classification rules  (first match wins — order matters) ─────────────────
# Each tuple: (regex_on_folder_name_lowercase, destination_relative_to_root)
RULES: list[tuple[str, str]] = [

    # ── Duplicates (catch before anything else) ───────────────────────────────
    (r"- copy$",                                    "MISC/_duplicates"),

    # ── DOD / Department of War ───────────────────────────────────────────────
    # Range Fouler reports (name contains "range-fouler" or "range_fouler")
    (r"dow-uap-d\d+.*range.fouler.*arabian",        "DOD/range-fouler-debriefs/arabian"),
    (r"dow-uap-d\d+.*range.fouler.*japan",          "DOD/range-fouler-debriefs/japan"),
    (r"dow-uap-d\d+.*range.fouler.*aden",           "DOD/range-fouler-debriefs/gulf-of-aden"),
    (r"dow-uap-d\d+.*range.fouler.*middle",         "DOD/range-fouler-debriefs/middle-east"),
    (r"dow-uap-d\d+.*range.fouler",                 "DOD/range-fouler-debriefs/other"),

    # Email correspondence
    (r"dow-uap-d\d+.*email",                        "DOD/email-correspondence"),

    # Mission reports by region
    (r"dow-uap-d\d+.*mission.*arabian",             "DOD/mission-reports/arabian-gulf"),
    (r"dow-uap-d\d+.*mission.*iraq",                "DOD/mission-reports/iraq"),
    (r"dow-uap-d\d+.*mission.*syria",               "DOD/mission-reports/syria"),
    (r"dow-uap-d\d+.*mission.*persian",             "DOD/mission-reports/persian-gulf"),
    (r"dow-uap-d\d+.*mission.*hormuz",              "DOD/mission-reports/strait-of-hormuz"),
    (r"dow-uap-d\d+.*mission.*greece",              "DOD/mission-reports/greece"),
    (r"dow-uap-d\d+.*mission.*emirates",            "DOD/mission-reports/uae"),
    (r"dow-uap-d\d+.*mission.*aden",                "DOD/mission-reports/gulf-of-aden"),
    (r"dow-uap-d\d+.*mission.*mediterranean",       "DOD/mission-reports/mediterranean"),
    (r"dow-uap-d\d+.*mission.*iran",                "DOD/mission-reports/iran"),
    (r"dow-uap-d\d+.*mission.*djibouti",            "DOD/mission-reports/djibouti"),
    (r"dow-uap-d\d+.*mission.*southern",            "DOD/mission-reports/southern-us"),
    (r"dow-uap-d\d+.*mission.*middle",              "DOD/mission-reports/middle-east"),
    (r"dow-uap-d\d+.*mission.*china",               "DOD/mission-reports/east-china-sea"),
    (r"dow-uap-d\d+.*mission.*gulf.of.aden",        "DOD/mission-reports/gulf-of-aden"),
    (r"dow-uap-d\d+.*mission",                      "DOD/mission-reports/other"),

    # DOD catch-alls
    (r"dow-uap-pr\d+",                              "DOD/reports-other"),
    (r"dow-uap-d\d+",                               "DOD/reports-other"),

    # ── NASA ──────────────────────────────────────────────────────────────────
    (r"nasa-uap-d\d+.*transcript",                  "NASA/transcripts"),
    (r"nasa-uap-d\d+.*debriefing",                  "NASA/crew-debriefings"),
    (r"nasa-uap",                                   "NASA/other"),

    # ── FBI ───────────────────────────────────────────────────────────────────
    (r"fbi-photo",                                  "FBI/photo-collections"),

    # ── Department of State ───────────────────────────────────────────────────
    (r"dos-uap",                                    "DOS/cables"),

    # ── NARA / CIA archives ───────────────────────────────────────────────────
    (r"65_hs1-834228961",                           "NARA-CIA/hs1-834228961"),
    (r"65_hs1-101634279",                           "NARA-CIA/hs1-101634279"),
    (r"^341_",                                      "NARA-CIA/series-341"),
    (r"^342_",                                      "NARA-CIA/series-342"),
    (r"^331_",                                      "NARA-CIA/series-331"),
    (r"^38_",                                       "NARA-CIA/series-38"),
    (r"^59_",                                       "NARA-CIA/series-59"),
    (r"^18_",                                       "NARA-CIA/series-18"),
    (r"^255_",                                      "NARA-CIA/series-255"),

    # ── MISC ──────────────────────────────────────────────────────────────────
    (r"serial.*redacted|usper.*statement",          "MISC/statements-redacted"),
    (r"sketch|composite",                           "MISC/visuals"),
    (r"slides",                                     "MISC/visuals"),
    (r"059uap",                                     "MISC/unclassified"),
    (r"western_us",                                 "MISC/presentations"),
]

# Folders (and files) that should never be moved
SKIP_NAMES = {
    "reorganize.py",
    "reconcile.py",
    "pdf_to_reports.py",
    "uap_record_schema.yaml",
    "uap-csv.csv",
    "move_log.json",
    "records",
    "reports_out",
    "MISC",
    "DOD",
    "NASA",
    "FBI",
    "DOS",
    "NARA-CIA",
    "Untitled.md",
}


# ── core logic ────────────────────────────────────────────────────────────────

def classify(name: str) -> str | None:
    """Return destination subpath (relative to root) for a folder name, or None to skip."""
    lower = name.lower()
    for pattern, dest in RULES:
        if re.search(pattern, lower):
            return dest
    return None


def plan_moves(root: Path) -> list[dict]:
    """
    Walk the immediate children of root (folders only) and return a list of
    planned move operations: {src, dst, dest_category}.
    """
    moves = []
    unmatched = []

    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        if item.name in SKIP_NAMES:
            continue

        dest_rel = classify(item.name)
        if dest_rel is None:
            unmatched.append(item.name)
            continue

        dst_dir = root / dest_rel
        dst     = dst_dir / item.name

        moves.append({
            "src":           str(item),
            "dst":           str(dst),
            "dest_category": dest_rel,
        })

    return moves, unmatched


def print_plan(moves: list[dict], unmatched: list[str]) -> None:
    current_cat = None
    for m in moves:
        cat = m["dest_category"]
        if cat != current_cat:
            print(f"\n  📁  {cat}/")
            current_cat = cat
        src_name = Path(m["src"]).name
        print(f"       {src_name}")

    if unmatched:
        print(f"\n  ⚠  {len(unmatched)} folder(s) did not match any rule:")
        for u in unmatched:
            print(f"       {u}")

    print(f"\n  Total: {len(moves)} moves planned,  {len(unmatched)} unmatched\n")


def execute_moves(moves: list[dict], root: Path, log_path: Path) -> None:
    done    = []
    errors  = []

    for m in moves:
        src = Path(m["src"])
        dst = Path(m["dst"])

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                print(f"  ⚠  destination exists, skipping: {dst.name}")
                continue
            shutil.move(str(src), str(dst))
            done.append(m)
            print(f"  ✓  {src.name}  →  {m['dest_category']}/")
        except Exception as exc:
            errors.append({"move": m, "error": str(exc)})
            print(f"  ✗  {src.name}  ERROR: {exc}")

    # Write undo log
    log = {
        "timestamp": datetime.now().isoformat(),
        "root":      str(root),
        "moves":     done,
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\n  ✅  {len(done)} moved,  {len(errors)} errors")
    print(f"  📝  Undo log written to {log_path}")


def undo_moves(log_path: Path) -> None:
    if not log_path.exists():
        print(f"  ✗  No undo log found at {log_path}")
        return

    log   = json.loads(log_path.read_text(encoding="utf-8"))
    moves = log.get("moves", [])

    print(f"  Reversing {len(moves)} moves from {log['timestamp']} …\n")
    errors = 0
    for m in reversed(moves):
        src = Path(m["dst"])   # swap: dst is where it ended up
        dst = Path(m["src"])   # src is where it came from
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  ↩  {src.name}  →  {dst.parent.name}/")
        except Exception as exc:
            print(f"  ✗  {src.name}  ERROR: {exc}")
            errors += 1

    print(f"\n  ✅  Undo complete.  {len(moves) - errors} restored,  {errors} errors")
    log_path.unlink(missing_ok=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Reorganise D:/divided into a logical folder hierarchy"
    )
    ap.add_argument("--root",    default=DEFAULT_ROOT, help="Root folder to reorganise")
    ap.add_argument("--execute", action="store_true",  help="Actually move folders (default: dry-run)")
    ap.add_argument("--undo",    action="store_true",  help="Reverse previous run using move_log.json")
    args = ap.parse_args()

    root     = Path(args.root)
    log_path = root / "move_log.json"

    if args.undo:
        undo_moves(log_path)
        return

    moves, unmatched = plan_moves(root)

    if not args.execute:
        print(f"\n{'─'*60}")
        print(f"  DRY RUN — nothing will be moved")
        print(f"  Root: {root}")
        print(f"{'─'*60}")
        print_plan(moves, unmatched)
        print("  Run with --execute to apply.\n")
    else:
        print(f"\n  Moving folders under {root} …\n")
        execute_moves(moves, root, log_path)

        if unmatched:
            print(f"\n  ⚠  {len(unmatched)} folder(s) left in place (no rule matched):")
            for u in unmatched:
                print(f"       {u}")


if __name__ == "__main__":
    main()
