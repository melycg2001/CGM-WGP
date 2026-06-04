"""
runs.py — CLI utility for managing CGM-WGP pipeline runs.

Commands:
  list       List all indexed runs (filterable by --crop, --tag, --label)
  tag        Add tags to a run
  label      Set a human-readable label on a run
  archive    Move old runs to pipeline/output/archive/
  backfill   Scan existing run directories and rebuild index.json
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from cgm_wgp.config import OUTPUT_DIR


INDEX_PATH = OUTPUT_DIR / "index.json"
ARCHIVE_DIR = OUTPUT_DIR / "archive"


def _load_index() -> list[dict]:
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_index(entries: list[dict]):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(entries, indent=2))


def cmd_list(args):
    entries = _load_index()
    if args.crop:
        entries = [e for e in entries if e.get("crop") == args.crop]
    if args.tag:
        entries = [e for e in entries if args.tag in e.get("tags", [])]
    if args.label:
        entries = [e for e in entries if args.label in (e.get("label") or "")]
    if args.cv_scheme:
        entries = [e for e in entries if e.get("cv_scheme") == args.cv_scheme]

    if not entries:
        print("No matching runs found.")
        return

    # Sort by created timestamp
    entries.sort(key=lambda e: e.get("created", ""))

    # Print table
    header = f"{'Created':<20} {'Crop':<12} {'Location':<14} {'Label':<24} {'Tags':<16} {'Metrics'}"
    print(header)
    print("-" * len(header))
    for e in entries:
        metrics_str = ""
        m = e.get("metrics", {})
        if m:
            parts = []
            if "ensemble_mae" in m:
                parts.append(f"MAE={m['ensemble_mae']}")
            elif "mech_mae" in m:
                parts.append(f"MAE={m['mech_mae']}")
            if "n_genotypes" in m:
                parts.append(f"n={m['n_genotypes']}")
            metrics_str = ", ".join(parts)
        tags_str = ",".join(e.get("tags", []))
        label_str = (e.get("label") or "")[:24]
        print(f"{e.get('created', '?'):<20} {e.get('crop', '?'):<12} "
              f"{e.get('location', '?'):<14} {label_str:<24} {tags_str:<16} {metrics_str}")

    print(f"\n{len(entries)} run(s)")


def cmd_tag(args):
    entries = _load_index()
    run_id = args.run_id
    tags_to_add = args.tags

    matched = False
    for entry in entries:
        if run_id in entry.get("created", "") or run_id in entry.get("path", ""):
            existing = entry.get("tags", [])
            for t in tags_to_add:
                if t not in existing:
                    existing.append(t)
            entry["tags"] = existing
            matched = True
            # Also update meta.json on disk
            run_path = OUTPUT_DIR / entry["path"]
            meta_path = run_path / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["tags"] = existing
                meta_path.write_text(json.dumps(meta, indent=2))
            print(f"Tagged {entry['path']} with: {tags_to_add}")

    if not matched:
        print(f"No run matching '{run_id}' found in index.", file=sys.stderr)
        sys.exit(1)

    _save_index(entries)


def cmd_label(args):
    entries = _load_index()
    run_id = args.run_id
    new_label = args.label

    matched = False
    for entry in entries:
        if run_id in entry.get("created", "") or run_id in entry.get("path", ""):
            entry["label"] = new_label
            matched = True
            # Also update meta.json on disk
            run_path = OUTPUT_DIR / entry["path"]
            meta_path = run_path / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["label"] = new_label
                meta_path.write_text(json.dumps(meta, indent=2))
            print(f"Labeled {entry['path']} as: {new_label}")

    if not matched:
        print(f"No run matching '{run_id}' found in index.", file=sys.stderr)
        sys.exit(1)

    _save_index(entries)


def cmd_archive(args):
    entries = _load_index()
    cutoff = args.before
    if not cutoff:
        print("--before YYYY-MM-DD is required", file=sys.stderr)
        sys.exit(1)

    to_archive = [e for e in entries if e.get("created", "9999") < cutoff]
    # Never archive tagged runs
    to_archive = [e for e in to_archive if not e.get("tags")]

    if not to_archive:
        print(f"No untagged runs before {cutoff} to archive.")
        return

    print(f"Archiving {len(to_archive)} runs older than {cutoff}...")
    if not args.yes:
        resp = input("Continue? [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived_paths = set()
    for entry in to_archive:
        src = OUTPUT_DIR / entry["path"]
        if src.exists():
            dest = ARCHIVE_DIR / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            archived_paths.add(entry["path"])
            print(f"  Moved: {entry['path']}")

    # Remove archived entries from index
    entries = [e for e in entries if e["path"] not in archived_paths]
    _save_index(entries)
    print(f"Archived {len(archived_paths)} runs. Index updated.")


def cmd_backfill(args):
    """Scan existing run directories and rebuild/extend index.json."""
    entries = _load_index()
    existing_paths = {e.get("path") for e in entries}

    # Scan both old (pipeline/output/{crop}/{loc}/{ts}) and new (pipeline/output/runs/{crop}/{loc}/{ts})
    scan_roots = [OUTPUT_DIR, OUTPUT_DIR / "runs"]
    new_count = 0

    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for crop_dir in sorted(scan_root.iterdir()):
            if not crop_dir.is_dir():
                continue
            # Skip non-crop directories
            if crop_dir.name in ("runs", "archive", "figures", "comparison_figures",
                                 "cross_crop_comparison", "publication_figures",
                                 "plots", "mech_comparison", "model_scheme_comparison",
                                 "cv_group_preview", "cv_preview", "unknown"):
                continue
            crop = crop_dir.name
            for loc_dir in sorted(crop_dir.iterdir()):
                if not loc_dir.is_dir():
                    continue
                if loc_dir.name == "latest" or loc_dir.is_symlink():
                    continue
                location = loc_dir.name
                for run_dir in sorted(loc_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    if run_dir.is_symlink():
                        continue
                    # Skip non-timestamp dirs (like loo_cv_*)
                    ts = run_dir.name
                    if not (len(ts) >= 10 and ts[4] == "-"):
                        continue

                    rel_path = str(run_dir.relative_to(OUTPUT_DIR))
                    if rel_path in existing_paths:
                        continue

                    # Build entry from meta.json or run_config.json
                    meta = {"crop": crop, "location": location, "created": ts,
                            "label": None, "tags": [], "git_sha": "", "metrics": {},
                            "path": rel_path}

                    # Try reading meta.json (new format)
                    meta_file = run_dir / "meta.json"
                    if meta_file.exists():
                        try:
                            disk_meta = json.loads(meta_file.read_text())
                            meta.update({k: disk_meta[k] for k in disk_meta if k in meta})
                        except Exception:
                            pass

                    # Try extracting metrics from legacy files
                    for legacy_name in ("alpha_beta_results_metrics_summary.csv",):
                        f = run_dir / legacy_name
                        if f.exists() and not meta["metrics"]:
                            try:
                                import pandas as pd
                                df = pd.read_csv(f)
                                if "mae" in df.columns:
                                    meta["metrics"]["mech_mae"] = round(float(df["mae"].mean()), 2)
                            except Exception:
                                pass

                    entries.append(meta)
                    existing_paths.add(rel_path)
                    new_count += 1

    _save_index(entries)
    print(f"Backfill complete. Added {new_count} new entries. Total: {len(entries)} runs indexed.")


def main():
    p = argparse.ArgumentParser(description="Manage CGM-WGP pipeline runs")
    sub = p.add_subparsers(dest="command")

    # list
    ls = sub.add_parser("list", help="List indexed runs")
    ls.add_argument("--crop", default=None)
    ls.add_argument("--tag", default=None)
    ls.add_argument("--label", default=None)
    ls.add_argument("--cv-scheme", default=None)

    # tag
    tg = sub.add_parser("tag", help="Add tags to a run")
    tg.add_argument("run_id", help="Run timestamp or path substring to match")
    tg.add_argument("tags", nargs="+", help="Tags to add")

    # label
    lb = sub.add_parser("label", help="Set a label on a run")
    lb.add_argument("run_id", help="Run timestamp or path substring to match")
    lb.add_argument("label", help="Label text")

    # archive
    ar = sub.add_parser("archive", help="Archive old runs")
    ar.add_argument("--before", required=True, help="Archive runs before this date (YYYY-MM-DD)")
    ar.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # backfill
    sub.add_parser("backfill", help="Scan existing directories and rebuild index.json")

    args = p.parse_args()
    if not args.command:
        p.print_help()
        sys.exit(1)

    cmds = {"list": cmd_list, "tag": cmd_tag, "label": cmd_label,
            "archive": cmd_archive, "backfill": cmd_backfill}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
