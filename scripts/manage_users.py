"""Admin tool: list users and change account roles.

Clinician accounts are NOT self-service - anyone can register, but only an administrator with access to the
server can promote an account to "clinician" (after verifying who they are).

    python scripts/manage_users.py list
    python scripts/manage_users.py make-clinician dr_smith
    python scripts/manage_users.py make-patient dr_smith

Uses the same database as the app (SKINSENSE_DB / SKINSENSE_DATABASE_URL are respected).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from sqlalchemy import func  # noqa: E402

import database  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "make-clinician", "make-patient"])
    ap.add_argument("username", nargs="?")
    args = ap.parse_args()

    database.init_db()
    with database.SessionLocal() as db:
        if args.action == "list":
            for u in db.query(database.User).order_by(database.User.id).all():
                n = db.query(database.History).filter(database.History.user_id == u.id).count()
                print(f"{u.id:>4}  {u.username:<24} {u.role or 'patient':<10} {n} scans")
            return
        if not args.username:
            sys.exit("A username is required.")
        user = db.query(database.User).filter(func.lower(database.User.username) == args.username.lower()).first()
        if not user:
            sys.exit(f"No such user: {args.username}")
        user.role = "clinician" if args.action == "make-clinician" else "patient"
        db.commit()
        print(f"{user.username} is now a {user.role}.")


if __name__ == "__main__":
    main()
