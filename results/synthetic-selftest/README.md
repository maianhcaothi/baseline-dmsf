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
   reports (3 edge + 2 cloud, one cluster) — **all fourteen** result files plus
   the two accuracy files.
3. Free time goes through the real `FreeTimeTracker`, so the union, the exact
   `busy + free == span`, and the priority-ordered reason attribution are
   computed rather than fabricated. The accuracy pair goes through the real
   metric on invented boxes: a real computation over a fake detection set.
4. `guide/validate_results.py` reports **14/14 result files present, 6/6
   required** with **0 errors, 0 warnings**, and a deliberately corrupted copy is
   caught.
5. The notebook renders its charts into `imgs/` with zero cell errors.

## Replacing it with a real run

The server archives each run to `<log-path>/results/results_<MMDD>_<HHMM>_<tag>/`.
The notebook picks the newest directory containing `batch_done_ns.log`, so once a
real run exists this fixture stops being selected. Delete it once that happens:

```bash
python guide/validate_results.py <real-run-dir> --names cluster
python build_nb.py && python run_nb.py
```

Regenerate this fixture in place with the harness that sits beside it:

```bash
python results/synthetic-selftest/make_fixture.py
```
