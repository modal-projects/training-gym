"""Ad hoc dev-set spot check the agent wrote instead of always going
through toolbox/eval_toolbox/rubric_eval.py (fixture: an invented,
non-seed tool, used twice in this demo run).
"""
import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    rows = json.loads(open(args.answers).read())
    print(f"quick probe: {len(rows)}/{len(rows)} candidates present")


if __name__ == "__main__":
    main()
