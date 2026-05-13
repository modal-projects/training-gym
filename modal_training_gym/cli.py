"""CLI entry point: ``training-gym <command>``."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="training-gym")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="Deploy the training-gym dashboard to Modal")

    args = parser.parse_args()

    if args.command == "setup":
        from modal_training_gym.setup import setup

        setup()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
