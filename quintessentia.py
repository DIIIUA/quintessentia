"""Quintessentia: `python quintessentia.py <doctor|ask|chat|serve|discover|models|download|prepare> [options]`."""
import sys

sys.dont_write_bytecode = True

from quintessentia_cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
