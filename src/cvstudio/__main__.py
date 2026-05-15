"""Entry point. By default launches the PySide6 UI; `--list` prints the
registered operations and exits (the previous pre-alpha behaviour)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from cvstudio import __version__
from cvstudio.core.registry import all_operations
from cvstudio.operations import load_builtin_operations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cvstudio", description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print registered operations and exit (no UI).",
    )
    parser.add_argument("--version", action="version", version=f"cvstudio {__version__}")
    args = parser.parse_args(argv)

    if args.list:
        load_builtin_operations()
        ops = all_operations()
        print(f"cvstudio v{__version__} — {len(ops)} operation(s) registered")
        for op in ops:
            print(f"  [{op.category}] {op.name:<20s}  {op.id}")
        return 0

    from cvstudio.ui.app import run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
