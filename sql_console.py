"""Tiny SQL console for the SkinSense database.

READ-ONLY by default (the connection is opened in SQLite read-only mode).
Pass --write to allow changes; a confirmation is required for every
statement that is not a SELECT.

    python sql_console.py            # safe browsing
    python sql_console.py --write    # allow UPDATE / DELETE / ...
"""
import argparse
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_db():
    env = os.environ.get("SKINSENSE_DB")
    for p in (env, HERE / "backend" / "skinsense.db", HERE / "skinsense.db"):
        if p and Path(p).exists():
            return Path(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="allow modifying statements (asks to confirm)")
    args = ap.parse_args()

    db_path = find_db()
    if db_path is None:
        print("Error: database not found. Start the backend first.")
        return

    uri = f"file:{db_path.as_posix()}?mode={'rw' if args.write else 'ro'}"
    conn = sqlite3.connect(uri, uri=True)
    print("\n" + "=" * 30)
    print(f" SKIN SENSE AI - SQL CONSOLE ({'WRITE' if args.write else 'read-only'})")
    print("=" * 30)
    print("Type 'exit' to quit. Password hashes are hidden from output.\n")

    while True:
        try:
            query = input("sql> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break

            is_select = query.lower().startswith(("select", "pragma", "explain"))
            if not is_select:
                if not args.write:
                    print("Read-only mode: restart with --write to modify data.")
                    continue
                if input(f"Run modifying statement?  {query}\n  type 'yes' to confirm: ").strip().lower() != "yes":
                    print("Cancelled.")
                    continue

            cur = conn.cursor()
            cur.execute(query)
            if is_select:
                rows = cur.fetchall()
                if not rows:
                    print("No results found.")
                    continue
                cols = [d[0] for d in cur.description]
                hide = {i for i, c in enumerate(cols) if "password" in c.lower()}
                print(" | ".join(cols))
                print("-" * (len(" | ".join(cols)) + 4))
                for row in rows:
                    print(" | ".join("<hidden>" if i in hide else str(v) for i, v in enumerate(row)))
            else:
                conn.commit()
                print(f"Success: {cur.rowcount} row(s) affected.")
        except sqlite3.Error as exc:
            print(f"SQL Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            break

    conn.close()
    print("\nConsole closed.")


if __name__ == "__main__":
    main()
