# HLADMM for Boolean quadratic programming with generalized upper bound (GUB) constraints

Python code, instance generator and result scripts accompanying the paper

> H. Sharma and R. K. Nayak, *A Hybrid Linearized ADMM for Boolean Quadratic Programming with
> Generalized Upper Bound Constraints*, submitted to *Optimization Letters* (under revision).

The code solves

```
max  f(x) = p'x + x'Rx     s.t.  sum_{j in C_k} x_j = 1  (k = 1..m),   x_j in {0,1}
```

for a partition {C_1,...,C_m} of {1,...,n} (BQP-GUB; the cluster-restricted maximum induced subgraph
problem, CMISP, is an instance of it).

## Method in one paragraph

The binary problem is relaxed to a product of simplices and split with `z`. For `g(x) = -p'x - x'Rx` the
continuous iteration (order z, x, lambda; `x` unconstrained) is

```
z^{t+1}      = Proj_Delta( x^t + lambda^t / rho )                      (block-wise simplex projection)
x^{t+1}      = ( eta x^t + rho z^{t+1} - grad g(x^t) - lambda^t ) / (eta + rho)
lambda^{t+1} = lambda^t + rho ( x^{t+1} - z^{t+1} )
```

with `L = ||R + R'||_2`, `eta = kappa L`, `rho = c L`. The paper proves convergence of this continuous
iteration to a stationary point of the relaxation when `delta(eta, rho) > 0` (condition (A); for
`kappa = 1.1` this means `c > 4.18`). A binary solution is then recovered by block-wise rounding of `z`,
best-improvement one-flip local search inside the blocks, and `R_max` random restarts with an elite pool.
**The convergence result does not cover the binary solution.**

## Repository contents

| File | Purpose |
|---|---|
| `hladmm_final.py` | The solver used for **all result tables** of the paper, plus instance reader and experiment runner. |
| `hladmm.py` | Library version used by `ablation.py` and `sweep.py` (adds the ablation variants: projected gradient, no ADMM, time-matched random starts, guided roundings). Run with `L_safety=1.0` it gives the same results as `hladmm_final.py` for the same seed (checked on the instances of the ablation). |
| `gen_instances.py` | Seeded instance generator (writes compact `.npz` files). |
| `exact_enum.py` | Exact optimum by complete enumeration of all `prod_k |C_k|` feasible selections (small instances only). |
| `ablation.py` | Ablation study with several seeds per instance. |
| `sweep.py` | Optional sensitivity run over `c = rho/L`. |
| `make_table.py` | Merges result CSVs with the Knitro and exact values and prints the summary statistics. |
| `make_latex.py` | Writes the computational tables of the paper in LaTeX (`booktabs`, `longtable`). |
| `make_ablation_table.py` | Writes the ablation table in LaTeX. |
| `knitro_values.csv` | Knitro objective values from the earlier runs (see "Limitations"). |
| `exact_values.csv` | Exact optima (columns `name,exact`) for the instances where they are known. |

Do not use any earlier prototype of the solver (for example one that stops after a single small step);
`hladmm_final.py` implements Algorithm 1 of the paper (stopping after two consecutive small steps).

## Requirements

* Python >= 3.10 (the paper's runs used Python 3.14), NumPy, SciPy.
* pandas for `make_latex.py` and `make_ablation_table.py`.
* LaTeX packages `booktabs` and `longtable` for the generated tables.

```
pip install numpy scipy pandas
```

Memory: the solver keeps `R` and `S = R + R'` as dense matrices; the largest instance (`n = 10,000`)
needs roughly 2-3 GB of RAM.

## Instances

Instance `n`-`k` of a family is looked up as `<prefix>-<n>-<k>` with extension `.txt`, `.mat` or `.npz`
(in this order) in the folder given by `--dir`. Prefixes: `bqpgubs` (also `bqpgub`) and `cmisp`.
**If several extensions exist, the first one is used, so keep only one file per instance.**

| Set | n | Instances | Source |
|---|---|---|---|
| Original BQP-GUB | 20, 50, 100, 200, 500, 1000, 2000 | 5 each | `p_i, r_ij` uniform on [-50, 50] following Wang and Punnen (2017); generated **without a recorded seed**; the archived files are the reference for all Knitro comparisons. Irregular blocks (3 to 524 elements). |
| Generated BQP-GUB | 2500, 3000, 4000, 5000, 6000, 7000 (5 each), 8000, 10000 (3 each) | `gen_instances.py`, seed 12345 | Uniform entries on [-50, 50], random partition into blocks of size 5 (`m = n/5`). |
| CMISP | 20, 50, 100, 200, 500, 1000, 2000 | 5 each | Irregular clusters for `n <= 200`, clusters of size 20 for `n >= 500`. |

Data location: `<REPOSITORY / ZENODO DOI WITH THE INSTANCE FILES>`.

### File formats

`.txt` (BQP-GUB): lines `R = [ ... ]` (rows separated by `;`, may span several lines), `p = [ ... ]`,
optionally `m = k`, and one line `C1 = [ ... ]`, `C2 = [ ... ]`, ... with **1-based** indices.
CMISP `.txt`: the matrix rows come first (one per line), followed by `p = ...` and the `C_k` lines.
`.mat`: variables `R`, `p` and a cell array `groups` (1-based). `.npz`: arrays `R`, `p`, `sizes`
(block sizes) and `perm` (0-based index order; blocks are consecutive pieces of `perm`).

## Quick start

```
python gen_instances.py --dims 200 --counts 2 --bs 5 --dist uniform --perm --dir data
python hladmm_final.py bqpgub --dims 200 --counts 2 --dir data --restarts 20 --tol 0.001 --out test.csv
```

Command-line options of `hladmm_final.py`:

| Option | Default in the script | Value used in the paper |
|---|---|---|
| `--kappa` | 1.1 | 1.1 (`eta = kappa L`) |
| `--c` | 5.0 | 5 (`rho = c L`; condition (A) needs `c > 4.18`) |
| `--max-iter` | 2000 | 2000 |
| `--tol` | 0.01 | **0.001** (two consecutive steps with `||x^{t+1}-x^t|| <= tol`) |
| `--restarts` | 5 | **20** |
| `--elite` | 5 | 5 |
| `--seed` | 0 | 0 |

The defaults for `--tol` and `--restarts` are **not** the settings of the paper: always pass
`--tol 0.001 --restarts 20` (or change the defaults).

### Output columns (`--out`)

`Filename, n, m, minCk, maxCk, HLADMM, Iter, F-Violation, P-Resi, D-Resi, Time, Starts-at-limit, L, eta, rho, condA`

`HLADMM` is the objective value of the best **binary** solution. `Iter`, `F-Violation`, `P-Resi`
(`||x - z||`) and `D-Resi` (`rho ||z^{t+1} - z^t||`) refer to the start that produced the best value;
`Starts-at-limit` counts the starts (out of `R_max`) that stopped at `--max-iter`; `condA` states whether
`delta(eta, rho) > 0`. `Time` is the wall-clock time of the whole solver call (all restarts, including the
computation of `L`, excluding file input). The solver checks that the returned solution is binary,
satisfies the GUB constraints and has the reported objective value.

## Reproducing the paper

All commands use `--dir data` for the instance folder.

**1. Generated large instances** (seed 12345):

```
python gen_instances.py --dims 2500 3000 4000 5000 6000 7000 --counts 5 5 5 5 5 5 --bs 5 --dist uniform --perm --dir data
python gen_instances.py --dims 8000 10000 --counts 3 3 --bs 5 --dist uniform --perm --dir data
```

**2. Main results** (one run per instance, seed 0, 20 starts):

```
python hladmm_final.py bqpgub --dims 20 50 100 200 500 1000 2000 --counts 5 5 5 5 5 5 5 --dir data --restarts 20 --tol 0.001 --out final_bqpgub_small.csv
python hladmm_final.py cmisp  --dims 20 50 100 200 500 1000 2000 --counts 5 5 5 5 5 5 5 --dir data --restarts 20 --tol 0.001 --out final_cmisp.csv
python hladmm_final.py bqpgub --dims 2500 3000 4000 5000 6000 7000 --counts 5 5 5 5 5 5 --dir data --restarts 20 --tol 0.001 --out final_bqpgub_large.csv
python hladmm_final.py bqpgub --dims 8000 10000 --counts 3 3 --dir data --restarts 20 --tol 0.001 --out final_bqpgub_xl.csv
```

The large runs take about 2.5 hours in total on a laptop (median 25 s for `n = 2500`, 676 s for
`n = 10,000`). Run them one at a time; two simultaneous jobs distort the timings.

**3. Exact optima** (complete enumeration; instances with more than `--max_enum` feasible points are skipped):

```
python exact_enum.py bqpgub --dims 20 --counts 5 --dir data --out exact_bqpgub.csv
python exact_enum.py cmisp  --dims 20 50 --counts 5 5 --dir data --out exact_cmisp.csv
```

`exact_values.csv` (columns `name,exact`) contains the values used in the paper.

**4. Ablation** (10 seeds, 20 starts; uses `hladmm.py` with `L_safety = 1.0`):

```
python ablation.py bqpgub --dims 200 500 1000 --counts 5 5 3 --seeds 10 --restarts 20 --tol 0.001 --out abl_bqpgub_v2.csv
python ablation.py cmisp  --dims 200 500 --counts 5 5 --seeds 10 --restarts 20 --tol 0.001 --out abl_cmisp_v2.csv
python ablation.py bqpgub --dims 5000 --counts 3 --seeds 3 --restarts 20 --tol 0.001 --out abl_5000_v2.csv
```

Variants compared with the full method: rounding only, no ADMM (random starts, same number of starts),
single start, projected gradient instead of ADMM, relaxation-guided random roundings, and random starts
followed by the same local search for the wall-clock time of the full method. The seed-0 value of the full
method equals the value in the main tables. One ablation run takes several times the time of one
`hladmm_final.py` run.

**5. Tables:**

```
python make_table.py final_bqpgub_small.csv final_bqpgub_large.csv final_bqpgub_xl.csv final_cmisp.csv --knitro knitro_values.csv --exact exact_values.csv --out merged_final.csv
python make_latex.py merged_final.csv tables.tex
python make_ablation_table.py --bqpgub abl_bqpgub_v2.csv --cmisp abl_cmisp_v2.csv --large abl_5000_v2.csv --out ablation_table.tex
```

`make_table.py` also prints the counts better / equal / worse against Knitro, the number of starts at the
iteration limit and whether condition (A) holds in every run.

`<TODO: add the command/script that produced the block-size experiment (Table 8 of the paper)>`

## Limitations to keep in mind

* The **Knitro values** (`knitro_values.csv`) come from earlier runs on another machine under a licence that has
  expired. They cannot be reproduced, so no timing comparison is made. Configuration:
  `<Knitro version, algorithm, tolerances, whether the continuous relaxation was solved>`.
* The original BQP-GUB instances were generated without a seed; use the archived files, not a regenerated set.
* For `n >= 1000` most starts stop at the iteration limit, so the reported values depend on `--max-iter`.
  The convergence theorem is asymptotic and does not claim that these runs reached the stationary point.
* The ablation shows that the local search and the number of restarts determine most of the quality of the
  binary solution: random restarts with the same local search, given the same wall-clock time, match or exceed
  the full method. The convergence guarantee concerns only the continuous relaxation.
* No comparison with the tabu-search methods of Wang and Punnen (2017) or Shang et al. (2023) is included.
* Timings of single instances can vary with other load on the machine; medians are more reliable.

## Citation

```
@article{sharma_nayak_hladmm,
  author  = {Sharma, Hitarth and Nayak, Rupaj Kumar},
  title   = {A Hybrid Linearized {ADMM} for Boolean Quadratic Programming with Generalized Upper Bound Constraints},
  journal = {Optimization Letters},
  note    = {Submitted},
  year    = {2026}
}
```

## Licence and contact

`<LICENCE, for example MIT>`. Contact: Hitarth Sharma (hitarthsharma321@gmail.com),
Rupaj Kumar Nayak (rupaj@iiit-bh.ac.in), IIIT Bhubaneswar.[README.md](https://github.com/user-attachments/files/33899851/README.md)
