"""Verify Supabase connectivity after applying 002_assignment_schema.sql.

Usage: python -m db.seed
"""
import os

from dotenv import load_dotenv

from api import db

load_dotenv()


def main() -> None:
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY first.")
    runs = db.list_runs(limit=5)
    print(f"Connected. runs table reachable; {len(runs)} recent run(s).")


if __name__ == "__main__":
    main()