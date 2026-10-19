"""Push saved run JSONs (weights/*_run.json) to the Supabase `runs` table.

    python log_runs.py                              # every weights/*_run.json
    python log_runs.py weights/baseline_run.json    # just one

Skips a run whose `name` already exists in the table unless --force is given.
Reads SUPABASE_URL and SUPABASE_SERVICE_KEY from .env or the shell.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (in .env or the shell).")

    files = [Path(f) for f in args.files] or sorted(Path("weights").glob("*_run.json"))
    if not files:
        sys.exit("No run JSON files found.")

    sb = create_client(url, key)
    for f in files:
        row = json.loads(f.read_text())
        existing = sb.table("runs").select("id").eq("name", row["name"]).execute().data
        if existing and not args.force:
            print(f"SKIP {row['name']}: already in runs (id={existing[0]['id']}). Use --force to insert anyway.")
            continue
        res = sb.table("runs").insert(row).execute()
        print(f"OK   {row['name']}: runs.id = {res.data[0]['id']}")


if __name__ == "__main__":
    main()