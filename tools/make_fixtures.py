"""Regenerate ``tests/fixtures/r_reference.json`` by running R.

Requires R with packages jsonlite, Formula, Matrix, mgcv (and splines).

    python tools/make_fixtures.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import rcases  # noqa: E402


def main():
    work = tempfile.mkdtemp(prefix="statsformula_r_")
    try:
        datasets = rcases.make_datasets()
        schema = {}
        for name, df in datasets.items():
            df.to_csv(os.path.join(work, name + ".csv"), index=False, na_rep="NA")
            schema[name] = rcases.schema(df)
        with open(os.path.join(work, "schema.json"), "w") as fh:
            json.dump(schema, fh)
        with open(os.path.join(work, "cases.json"), "w") as fh:
            json.dump(rcases.ALL_CASES, fh)
        subprocess.run(["Rscript", os.path.join(HERE, "r_reference.R"), work], check=True)
        out = os.path.join(ROOT, "tests", "fixtures", "r_reference.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(os.path.join(work, "r_reference.json")) as fh:
            results = json.load(fh)
        r_version = subprocess.run(["Rscript", "-e", "cat(R.version.string, as.character(packageVersion('mgcv')), as.character(packageVersion('Formula')), as.character(packageVersion('Matrix')))"],
                                   capture_output=True, text=True).stdout
        with open(out, "w") as fh:
            json.dump({"r_version": r_version, "cases": rcases.ALL_CASES, "results": results}, fh)
        print("wrote", out, len(results), "cases;", r_version)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
