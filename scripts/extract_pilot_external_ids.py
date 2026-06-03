"""
Extract external user IDs from pilot SQLite databases into a text file.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan pilot SQLite databases and write external user IDs "
            "to a newline-separated text file."
        )
    )
    parser.add_argument(
        "--pilots-dir",
        default="pilots",
        help="Directory containing pilot SQLite databases.",
    )
    parser.add_argument(
        "--output",
        default="pilots/external_ids.txt",
        help="Output text file path.",
    )
    return parser.parse_args()


def list_database_paths(pilots_dir: Path) -> list[Path]:
    """Return sorted SQLite database paths under the pilots directory."""
    return sorted(pilots_dir.glob("*.db"))


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Return True if the table exists in the database."""
    cursor = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    )
    return cursor.fetchone() is not None


def load_external_ids_from_db(db_path: Path) -> list[str]:
    """Load external IDs from a single SQLite database."""
    connection = sqlite3.connect(db_path)
    try:
        if not table_exists(connection, "externaluser"):
            return []
        cursor = connection.execute(
            "SELECT external_id FROM externaluser WHERE external_id IS NOT NULL"
        )
        return [row[0] for row in cursor.fetchall() if isinstance(row[0], str)]
    finally:
        connection.close()


def collect_external_ids(db_paths: Iterable[Path]) -> list[str]:
    """Collect unique external IDs from a list of SQLite databases."""
    valid_id = re.compile(r"^[a-z0-9]{24}$")
    seen: set[str] = set()
    for db_path in db_paths:
        for external_id in load_external_ids_from_db(db_path):
            if not valid_id.fullmatch(external_id):
                continue
            seen.add(external_id)
    return sorted(seen)


def write_external_ids(output_path: Path, external_ids: Iterable[str]) -> None:
    """Write external IDs to the output text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for external_id in external_ids:
            handle.write(f"{external_id}\n")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    pilots_dir = Path(args.pilots_dir)
    output_path = Path(args.output)
    db_paths = list_database_paths(pilots_dir)
    external_ids = collect_external_ids(db_paths)
    write_external_ids(output_path, external_ids)
    print(f"Wrote {len(external_ids)} external IDs to {output_path}")


if __name__ == "__main__":
    main()
