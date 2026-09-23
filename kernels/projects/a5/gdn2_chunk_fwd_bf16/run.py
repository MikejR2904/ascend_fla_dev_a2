"""Run the complete GDN-2 chunk unit with the canonical unit helper."""
from pathlib import Path

from _unit_runner import main


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
