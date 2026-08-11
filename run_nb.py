"""run_nb.py — execute the visualization notebook in place and report every failure.

`allow_errors=True` is the point: collect all failing cells in one pass instead
of one round-trip per bug.

    python build_nb.py && python run_nb.py
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# Same anchor as build_nb.py: this pair has to keep pointing at one notebook
# even after the project directory moves.
p = (Path(__file__).resolve().parent
     / "results" / "visual" / "DMSF Result Visualization.ipynb")
if not p.exists():
    sys.exit(f"notebook not found: {p}\nrun `python build_nb.py` first")
nb = nbformat.read(str(p), as_version=4)

NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(p.parent)}},
               allow_errors=True).execute()
nbformat.write(nb, str(p))

fail = 0
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            fail += 1
            print(f"\n### ERROR in cell {i} ###")
            print(c.source[:300], "\n---")
            print("\n".join(o.get("traceback", []))[-2500:])
        elif o.get("output_type") == "stream" and o.get("text", "").strip():
            print(f"[cell {i}] {o['text'].rstrip()[:1200]}")

print(f"\n=== {fail} cell error(s) ===")
sys.exit(1 if fail else 0)
