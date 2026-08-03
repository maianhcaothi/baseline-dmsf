# SYNTHETIC — not a measurement

Every number in this directory is **invented**. Nothing here was produced by
running the DMSF pipeline against a video, and none of it should be quoted,
compared against a real run, or put in a paper.

## Why it exists

To prove the result-format pipeline works end to end before there is a real run
to point it at:

1. `DmsfServer.on_fps` — the actual callback — was replayed against 430 synthetic
   arrivals, so `batch_done_ns.log` and `fps_cluster_ns.log` are written by the
   code the server will run, not by a stand-in.
2. The shutdown writers in `src/DmsfResults.py` were called with synthetic device
   reports (3 edge + 2 cloud, one cluster).
3. `guide/validate_results.py` passes with **0 errors, 0 warnings**, and a
   deliberately corrupted copy is caught.
4. The notebook renders all nine charts into `imgs/` with zero cell errors.

## Replacing it with a real run

The server archives each run to `<log-path>/results/results_<MMDD>_<HHMM>_<tag>/`.
The notebook picks the newest directory containing `batch_done_ns.log`, so once a
real run exists this fixture stops being selected. Delete it once that happens:

```bash
python guide/validate_results.py <real-run-dir> --names cluster
python build_nb.py && python run_nb.py
```

Regenerate this fixture with the harness in the scratchpad (`make_fixture.py`).
