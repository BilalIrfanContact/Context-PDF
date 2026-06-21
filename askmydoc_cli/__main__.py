from __future__ import annotations

from .app import run_cli


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
