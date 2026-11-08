"""Compare a benchmark run with its saved reference: the deployed (int8) student's test accuracy and the
labelling time. Exits with 1 when the accuracy fell by more than --tolerance or labelling took more than
--time-factor times as long, so a change that makes students worse or labelling slower is caught.

    python examples/benchmark/check.py REFERENCE.json RUN/results.json [--tolerance 0.01] [--time-factor 1.5]
"""

import argparse
import json
import sys


def facts(path: str) -> tuple[float, float]:
    r = json.load(open(path, encoding="utf-8"))
    models = {m["name"]: m for m in r["models"]}
    student = models.get("student, ONNX int8") or models["student, PyTorch"]
    return student["accuracy"], r["training"]["label_seconds"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("reference")
    p.add_argument("run")
    p.add_argument("--tolerance", type=float, default=0.01, help="accuracy it may lose (default 0.01)")
    p.add_argument(
        "--time-factor", type=float, default=1.5, help="labelling time it may take, times the reference's"
    )
    a = p.parse_args()
    ref_acc, ref_s = facts(a.reference)
    acc, s = facts(a.run)
    ok_acc, ok_time = acc >= ref_acc - a.tolerance, s <= ref_s * a.time_factor
    verdict_acc, verdict_time = ("ok" if ok_acc else "WORSE"), ("ok" if ok_time else "SLOWER")
    print(f"accuracy:  {acc:.3f} (reference {ref_acc:.3f}, may lose {a.tolerance})  {verdict_acc}")
    print(f"labelling: {s:.0f} s (reference {ref_s:.0f} s, at most x{a.time_factor})  {verdict_time}")
    sys.exit(0 if ok_acc and ok_time else 1)


if __name__ == "__main__":
    main()
