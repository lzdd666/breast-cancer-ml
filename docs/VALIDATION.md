# Validation record

Local validation completed on October 23, 2026, using Windows, Python 3.13.0,
and the package versions in `requirements.txt`.

- All eight unittest checks passed, covering class mapping, nested split
  isolation, training-only scaling, independence of inner selection from
  outer-test labels, metric/prediction consistency, within-environment
  repeatability, DBSCAN noise handling, and probability export precision.
- The complete 5-by-3 nested-CV experiment completed successfully. Its settings,
  selected parameters, predictions, source hashes and dataset fingerprint are
  included in `results/reference`.
- All 3,983 exported prediction rows were reloaded with round-trip float parsing.
  Recomputed per-fold metrics agreed with the saved metrics to within 1e-12.
  Each of the 569 samples appears exactly once per evaluated procedure.
- All inner split indices remain inside their outer training partition; no outer
  test index appears in the corresponding inner training or validation partitions.
- The scikit-learn dataset's feature values and remapped labels exactly match
  the supplied course `wdbc.data` file.
- The command-line smoke run completed. The three result figures were visually
  inspected, and relative README/report links were checked.
- The GitHub Actions workflow is configured for Windows and Ubuntu. It has not
  run on GitHub yet; local validation is not a claim of remote CI success.

The reference experiment does not establish statistical significance for the
small ensemble gain, independent external validity, or a clinical use case.
