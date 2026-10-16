# Standard APLR performance comparison

The comparison uses independently installed upstream and candidate wheels in
separate Python environments. Every fit runs in a fresh process, with imports
and synthetic data generation outside the fit timer. The benchmark reports fit
time, prediction time, CPU time, process peak resident memory, package versions, extension
hashes, aggregation diagnostics, and numerical differences.

All data is generated locally from seed zero. The default campaign covers
weighted Poisson/log and Tweedie/log fits with 10,000 and 50,000 rows, eight
predictors, and 0% or 75% exact duplicate feature rows. It uses explicit fixed
three-fold memberships, 100 boosting steps, 100 bins, and one fitting thread.
Each case has three repetitions for upstream, optimized APLR with aggregation
disabled, and optimized APLR with aggregation enabled.

```bash
git worktree add --detach ../aplr-baseline <upstream-base-commit>
python -m venv ../aplr-baseline/.venv
../aplr-baseline/.venv/bin/python -m pip install '../aplr-baseline[test]'
python -m venv .venv
.venv/bin/python -m pip install '.[test]'

../aplr-baseline/.venv/bin/python benchmarks/benchmark_standard.py \
  --baseline-python ../aplr-baseline/.venv/bin/python \
  --candidate-python .venv/bin/python \
  --suite parity --output benchmarks/parity-results.json

../aplr-baseline/.venv/bin/python benchmarks/benchmark_standard.py \
  --baseline-python ../aplr-baseline/.venv/bin/python \
  --candidate-python .venv/bin/python \
  --output benchmarks/standard-results.json
```

The parity suite also covers MSE/identity, binomial/logit, classification,
custom loss/link derivatives, and DataFrame preprocessing. It loads real
upstream model pickles in the candidate environment and verifies unchanged
predictions and retained time limits. Both suites require identical selected
term structures and compare coefficients, predictions, validation histories,
and CV error with `rtol=1e-10` and `atol=1e-12`.

Performance varies with hardware and the data distribution; timing is reported
rather than asserted in CI. Peak memory includes the interpreter, data,
fitted model, predictions, and retained comparison arrays. Identical-row
aggregation can cost time when few rows can be combined.
It retains original training rows for compatible final scoring, so fewer
working rows do not necessarily mean lower process peak memory.

## Results

The completed campaign contains 72 fits: eight loss/row-count/duplicate-rate
cases, three variants, and three repetitions per variant. The baseline is
upstream commit `d94685b64bfc7d94a0cb1a582f71d7751f56af99`; the candidate is
`e6187ab2b6a51d86b6336ba47be9cdbd56864595`. Both environments reported APLR
10.27.0, Python 3.12.3, NumPy 2.5.4, and pandas 3.0.6 on an AMD EPYC 7J13
64-Core Processor. The candidate source digest is recorded in
[`standard-results.json`](standard-results.json).

Each fit column is the median wall time in seconds over three fresh-process
repetitions. `Optimized` keeps `aggregate_identical_rows=False`, the default;
`Aggregated` enables it. Speedup is upstream median divided by the corresponding
candidate median, so values below 1.00x are slower than upstream.

| Loss | Rows | Exact duplicate rows | Upstream fit (s) | Optimized fit (s) | Optimized speedup | Aggregated fit (s) | Aggregated speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Poisson | 10,000 | 0% | 4.95 | 4.20 | 1.18x | 7.38 | 0.67x |
| Tweedie | 10,000 | 0% | 5.13 | 4.34 | 1.18x | 7.38 | 0.70x |
| Poisson | 10,000 | 75% | 4.83 | 3.97 | 1.22x | 4.68 | 1.03x |
| Tweedie | 10,000 | 75% | 5.18 | 4.41 | 1.17x | 5.01 | 1.03x |
| Poisson | 50,000 | 0% | 31.46 | 27.98 | 1.12x | 51.80 | 0.61x |
| Tweedie | 50,000 | 0% | 34.03 | 30.99 | 1.10x | 56.48 | 0.60x |
| Poisson | 50,000 | 75% | 25.07 | 21.61 | 1.16x | 25.54 | 0.98x |
| Tweedie | 50,000 | 75% | 29.46 | 26.21 | 1.12x | 31.57 | 0.93x |

The default optimized path was faster than upstream in all eight cases, with
per-case speedups from 1.10x to 1.22x (median 1.17x). Opt-in aggregation was
slower than the default optimized path in every case. With no duplicate rows,
aggregation ran at 0.60x–0.70x of upstream speed. With 75% duplicates, it was
about 1.03x upstream on the 10,000-row cases, 0.98x for 50,000-row Poisson,
and 0.93x for 50,000-row Tweedie. The measured row reduction therefore did not
translate to a fit-time win over the default optimized path.

Across the 24 fits per variant, median fit CPU time was 15.10 s upstream,
13.06 s optimized, and 16.48 s aggregated. Median prediction wall time was
10.20 ms, 9.68 ms, and 9.99 ms, respectively. Median process peak RSS was
153.6 MiB upstream, 183.3 MiB optimized, and 192.8 MiB aggregated. The
optimized path uses more memory for cached split data; aggregation also keeps
original training rows for exact final scoring, so its reduced working-row
count did not reduce peak process memory.

All 72 baseline comparisons passed, including selected-term structure and
numerical comparisons at `rtol=1e-10` and `atol=1e-12`; the largest recorded
absolute numerical difference was `4.44e-16`. The full per-fit evidence is in
[`standard-results.json`](standard-results.json). The separate seven-case
legacy/API parity suite results are recorded in
[`parity-results.json`](parity-results.json).
