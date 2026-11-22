#!/usr/bin/env python3
"""
fit_elf_ritchie_howie.py
========================
Read tabulated optical constants (n, k) from a CSV, plain-text or JSON file, build
the energy-loss function (ELF) Im[-1/eps(E)], fit it with a sum of Ritchie-Howie
(Drude-Lorentz type) oscillators, write the oscillator coefficients to a text file,
and plot the fit against the raw ELF.

Model (optical limit, q = 0)
----------------------------
    Im[-1/eps(E)] = sum_i  A_i * G_i * E / ( (E0_i^2 - E^2)^2 + (G_i * E)^2 )

    A_i  [eV^2]  oscillator strength / amplitude
    E0_i [eV]    oscillator position
    G_i  [eV]    oscillator damping (width)

ELF from n and k (eps = (n + ik)^2):
    Im[-1/eps] = eps2 / (eps1^2 + eps2^2) = 2nk / (n^2 + k^2)^2

If the ELF is already in your file (e.g. an extra column), it is ignored: the ELF
is always recomputed from n and k.

Input file
----------
Positional argument `input` (default: Sn_beta_Adachi_Table2.53.csv). The format is
chosen from the extension, or forced with --format.

  .csv     Comma-separated, with a header row. Needs columns for energy, n and k;
           other columns are ignored.
  .json    Any of
             {"E_eV": [...], "n": [...], "k": [...]}          dict of columns
             [{"E_eV": .., "n": .., "k": ..}, ...]            list of records
             [[E, n, k], ...]                                 list of rows (by position)
             {"columns": [...], "data": [[...], ...]}         pandas "split" layout
           optionally nested under another key such as "data".
  other    (.txt, .dat, ...) numbers separated by whitespace, tabs, commas or
           semicolons. Lines starting with '#' or '%' are comments. A header line
           (a non-numeric first line, or the last comment line before the data) is
           used if it contains energy, n and k names; otherwise the first three
           columns are taken as E [eV], n, k (a note is printed). Extra columns,
           e.g. a precomputed ELF, are ignored.

Column names are case-insensitive. The energy column may be called E_eV, E, eV,
energy, energy_eV, or with units, e.g. Energy[eV] or Energy (eV). The other
columns must be called n and k. Columns such as Wavelength[um] are ignored.

Several (n, k) column pairs: if a file has duplicate n/k columns (e.g. n, k, n, k),
then by default, for each row, the first pair with valid numbers is used (so pairs
covering different energy ranges are combined; the number of rows taken from each
pair is printed). Use --nk-set to select a single pair instead.

Point filtering (done before the fit; reported on the console):
  1. Rows with missing / non-numeric values are dropped.
  2. Rows with E <= 0, n <= 0 or k <= 0 are excluded. This removes negative n or
     k (non-physical noise in tabulated data) and zeros (ELF = 0 cannot be fitted
     in a log residual). The number of excluded points and their energy range
     are printed.
  3. --emin / --emax are then applied.
At least 5 points must remain.

Command-line options
--------------------
  input                 Input file (see above).

  --format {auto,csv,txt,json}
                        Input format. Default: auto (by extension: .json -> json,
                        .csv -> csv, anything else -> txt).

  --nk-set INDEX        Only for files with several (n, k) column pairs: use pair
                        number INDEX (0-based, in column order) instead of merging
                        them row by row.

  -N, --n-osc N         Number of oscillators. If omitted, it is asked for
                        interactively.

  --emin EMIN, --emax EMAX
                        Energy limits of the fit [eV]. Only data inside this
                        window are fitted and shown. Fitting a restricted window
                        around the feature of interest is often the easiest way to
                        control which part of the spectrum the oscillators describe.

  --mode {log,rel,abs,mix}
                        What is minimised (m = model, y = data):
                          log  (default) ln(m) - ln(y). Every point counts equally
                               in relative terms, so the tails (small ELF) matter
                               as much as the peaks. Good for describing the ELF
                               over many decades with enough oscillators; with
                               very few oscillators it can distort the main peak
                               (e.g. a single oscillator becomes too narrow).
                          rel  (m - y) / y. Similar to log for small deviations,
                               but asymmetric: an underestimate costs at most 1,
                               an overestimate is unbounded. Can end up in poor
                               local minima; usually log or mix is better.
                          abs  m - y. Dominated by the largest values, so it
                               reproduces the main peaks (position, width,
                               height) and the sum rule well, but the tails are
                               not followed. Best for fitting one or two dominant
                               peaks.
                          mix  (m - y) / (y + FLOOR * max(y)). Relative where the
                               ELF is small, absolute near the peaks: a compromise
                               between abs and log.

  --floor FLOOR         Only for --mode mix. ELF level, as a fraction of its
                        maximum, below which the residual becomes relative.
                        Default 0.05. Smaller values give more weight to the
                        tails (closer to log); larger values are closer to abs.

  --starts S            Number of optimisation starts (default 40). The first
                        start uses the ELF peak positions as initial E0 (G = 0.3*E0);
                        the others use random E0 (log-uniform over the data range)
                        and G = E0 * 10^U(-1.3, -0.1). Amplitudes are always
                        initialised by non-negative least squares. The best result
                        is kept. Use more starts (100+) for many oscillators.

  --seed SEED           Random seed for the starts (default 1); change it to test
                        the stability of the result.

  --max-nfev M          Maximum number of function evaluations per start
                        (default 3000).

  --atom-density RHO    Atomic density [cm^-3], used only for the effective
                        electrons-per-atom estimate from the f-sum rule (default
                        3.70e22, beta-Sn). Set it for your material, e.g. 4.99e22
                        for Si, otherwise the reported electrons per atom are wrong
                        (the fit itself is not affected).

  --out-prefix PREFIX   Prefix of the output files. Default: input file name
                        without extension + "_RH_N<N>".

  --ylin, --ylog        Force a linear / logarithmic y axis in the main plot.
                        Default: follows --mode, i.e. logarithmic for log and rel
                        (tails are fitted, so they should be visible), linear for
                        abs and mix (the peaks are what is fitted).

  --no-show             Do not open a plot window (the PNG is still saved);
                        useful on servers and in batch runs.

Fit details
-----------
The parameters are optimised in log space (A, E0, G > 0) with bounds
A in [1e-9, 1e9] eV^2, E0 in [Emin/2, 2*Emax], G in [1e-3, 10*Emax] eV, using
scipy.optimize.least_squares (trf). The oscillators are sorted by E0.

Output files
------------
  <prefix>_coefficients.txt   Header with the source file, model, number of
                              oscillators, fit range, residual mode, RMS and
                              maximum of |log10(fit/data)|, sum(A_i), the integral
                              of E*ELF (data and fit) and the effective number of
                              electrons per atom; then the table  i, A_i [eV^2],
                              E0_i [eV], Gamma_i [eV].
  <prefix>_fit.png            Top panel: data, fit and the individual oscillators
                              (log energy axis); bottom panel: fit/data ratio.

Sum rule: the f-sum rule gives  int_0^inf E*ELF(E) dE = (pi/2)*(hbar*omega_p)^2,
and for the model sum(A_i) = (hbar*omega_p)^2. The effective electrons per atom
are reported for the data in the fitted range, for the fit in the same range, and
for the fit in total (sum A_i). The latter can exceed the data value if the
model has substantial weight outside the fitted range.

Usage examples
--------------
    python fit_elf_ritchie_howie.py Sn_beta_Adachi_Table2.53.csv -N 8
    python fit_elf_ritchie_howie.py data.csv -N 10 --emin 1 --emax 5000 --starts 100
    python fit_elf_ritchie_howie.py data.csv            # asks for N interactively
    python fit_elf_ritchie_howie.py Si_n_k_Palik.txt -N 1 --mode abs --atom-density 4.99e22
    python fit_elf_ritchie_howie.py Si.json -N 4 --mode mix --floor 0.02 --nk-set 0 --no-show

Requires: numpy, pandas, scipy, matplotlib
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import least_squares, nnls
from scipy.signal import find_peaks

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
_E_BASES = {"e", "ev", "e_ev", "energy", "energy_ev", "photon energy", "photon_energy"}


def _is_energy(name):
    """True for names like E_eV, E, eV, energy, Energy[eV], Energy (eV), ..."""
    s = str(name).strip().lower()
    base = re.sub(r"[\[\(].*?[\]\)]", "", s).strip()
    return s in _E_BASES or base in _E_BASES


def _nk_columns(name):
    s = str(name).strip().lower()
    return "n" if s == "n" else "k" if s == "k" else None


def _read_text_table(path):
    """Read a whitespace-, comma- or semicolon-separated text table.

    Lines starting with '#' or '%' are comments; the last comment line before the
    data (or a non-numeric first line) is used as header if it contains E/n/k
    column names. Returns (DataFrame, positional_ok).
    """
    def split(s):
        return [t for t in re.split(r"[,;\s]+", s) if t]

    header, comment_header, rows = None, None, []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] in "#%":
                if not rows:
                    comment_header = split(s.lstrip("#% "))
                continue
            tokens = split(s)
            try:
                rows.append([float(t) for t in tokens])
            except ValueError:
                if not rows and header is None:
                    header = tokens            # non-numeric first line = header
    if not rows:
        sys.exit(f"No numeric data found in {path}.")
    lengths = [len(r) for r in rows]
    width = max(set(lengths), key=lengths.count)   # most common row length
    df = pd.DataFrame([r for r in rows if len(r) == width])

    names = header or comment_header
    if names and len(names) == width:
        low = {nm.strip().lower() for nm in names}
        if any(_is_energy(nm) for nm in names) and {"n", "k"} <= low:
            df.columns = names
    return df, True


def _json_to_frame(obj):
    """Turn a parsed JSON object into (DataFrame, positional_ok).

    Supported layouts:
      {"E_eV": [...], "n": [...], "k": [...]}            dict of columns
      [{"E_eV": .., "n": .., "k": ..}, ...]              list of records
      [[E, n, k], ...]                                   list of rows (positional)
      {"columns": [...], "data": [[...], ...]}           pandas 'split' layout
      any of the above nested under another key (e.g. {"data": ...})
    """
    if isinstance(obj, dict):
        if "columns" in obj and isinstance(obj.get("data"), list):
            return pd.DataFrame(obj["data"], columns=obj["columns"]), False
        if {"n", "k"} <= {str(k).lower() for k in obj}:
            return pd.DataFrame(obj), False
        for v in obj.values():
            if isinstance(v, (dict, list)):
                try:
                    return _json_to_frame(v)
                except (ValueError, TypeError):
                    continue
    elif isinstance(obj, list) and obj:
        if all(isinstance(r, dict) for r in obj):
            return pd.DataFrame(obj), False
        if all(isinstance(r, (list, tuple)) for r in obj):
            return pd.DataFrame(obj), True
    raise ValueError("unrecognised JSON structure")


def _read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        obj = json.load(f)
    try:
        return _json_to_frame(obj)
    except (ValueError, TypeError) as exc:
        sys.exit(f"Cannot interpret JSON file {path}: {exc}")


def _num(df, i):
    return pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy(float)


def _standardise(df, path, positional_ok, nk_set=None):
    """Return a numeric 3-column frame (E, n, k) from a table with arbitrary columns.

    If the table contains several (n, k) column pairs (duplicate names), then by
    default, for each row, the first pair with valid values is used; with
    nk_set=i only the i-th pair (0-based) is used.
    """
    names = [str(c).strip() for c in df.columns]
    e_idx = [i for i, c in enumerate(names) if _is_energy(c)]
    n_idx = [i for i, c in enumerate(names) if _nk_columns(c) == "n"]
    k_idx = [i for i, c in enumerate(names) if _nk_columns(c) == "k"]

    if e_idx and n_idx and k_idx:
        E = _num(df, e_idx[0])
        P = min(len(n_idx), len(k_idx))
        n_all = np.column_stack([_num(df, i) for i in n_idx[:P]])
        k_all = np.column_stack([_num(df, i) for i in k_idx[:P]])
        if P > 1:
            valid = ~np.isnan(n_all) & ~np.isnan(k_all)
            if nk_set is not None:
                if not 0 <= nk_set < P:
                    sys.exit(f"--nk-set {nk_set} out of range: {path} has {P} (n, k) pairs.")
                first = np.full(len(E), nk_set)
                print(f"Note: {path} has {P} (n, k) column pairs; using pair {nk_set}.")
            else:
                first = np.argmax(valid, axis=1)
                used = [int(np.sum(first[valid.any(axis=1)] == j)) for j in range(P)]
                print(f"Note: {path} has {P} (n, k) column pairs; per row the first pair "
                      f"with valid values is used (rows taken from each pair: {used}). "
                      f"Use --nk-set to pick one pair.")
            rows = np.arange(len(E))
            n, k = n_all[rows, first], k_all[rows, first]
        else:
            n, k = n_all[:, 0], k_all[:, 0]
        return pd.DataFrame({"E": E, "n": n, "k": k})

    if positional_ok and df.shape[1] >= 3:
        print(f"Note: no E/n/k column names found in {path}; "
              f"using the first three columns as E [eV], n, k.")
        return pd.DataFrame({"E": _num(df, 0), "n": _num(df, 1), "k": _num(df, 2)})
    raise KeyError(f"Columns E_eV, n, k not found in {path}. Available columns: {names}")


def load_elf(path, emin=None, emax=None, fmt="auto", nk_set=None):
    """Return E [eV] and ELF = Im[-1/eps] computed from n and k.

    fmt: 'csv', 'txt' (whitespace/comma/semicolon text), 'json' or 'auto'
    (by file extension: .json -> json, .csv -> csv, anything else -> txt).
    """
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = "json" if ext == ".json" else "csv" if ext == ".csv" else "txt"
    if fmt == "csv":
        df, positional_ok = pd.read_csv(path), False
    elif fmt == "json":
        df, positional_ok = _read_json(path)
    else:
        df, positional_ok = _read_text_table(path)

    df = _standardise(df, path, positional_ok, nk_set)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = ["E", "n", "k"]
    # Exclude non-physical points: negative (or zero) E, n or k. Zero is excluded too,
    # since ELF = 0 cannot be fitted in the log residual.
    bad = (df.E <= 0) | (df.n <= 0) | (df.k <= 0)
    if bad.any():
        print(f"Note: excluded {int(bad.sum())} point(s) with non-positive E, n or k "
              f"({int((df.n < 0).sum())} with n < 0, {int((df.k < 0).sum())} with k < 0, "
              f"{int(((df.n == 0) | (df.k == 0)).sum())} with n = 0 or k = 0, "
              f"{int((df.E <= 0).sum())} with E <= 0).")
        print(f"      E range of excluded points: {df.E[bad].min():.4g} ... {df.E[bad].max():.4g} eV")
    df = df[~bad].sort_values("E")
    if emin is not None:
        df = df[df.E >= emin]
    if emax is not None:
        df = df[df.E <= emax]
    if len(df) < 5:
        sys.exit("Too few valid data points after filtering.")

    E = df.E.to_numpy(float)
    n = df.n.to_numpy(float)
    k = df.k.to_numpy(float)
    elf = 2.0 * n * k / (n**2 + k**2) ** 2          # = eps2 / |eps|^2
    return E, elf


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
def basis(E, E0, G):
    """Matrix (len(E), N): contribution of each unit-amplitude oscillator."""
    E = np.asarray(E, float)[:, None]
    return G * E / ((E0**2 - E**2) ** 2 + (G * E) ** 2)


def elf_model(E, A, E0, G):
    return basis(E, E0, G) @ A


def unpack(theta):
    t = np.asarray(theta).reshape(-1, 3)
    return np.exp(t[:, 0]), np.exp(t[:, 1]), np.exp(t[:, 2])   # A, E0, G


def residuals(theta, E, y, mode, floor=0.05):
    A, E0, G = unpack(theta)
    m = elf_model(E, A, E0, G)
    if mode == "log":                      # deviation in ln(ELF)
        return np.log(m + 1e-300) - np.log(y)
    if mode == "rel":                      # relative deviation
        return (m - y) / y
    if mode == "mix":                      # relative for small ELF, absolute near the peaks
        return (m - y) / (y + floor * y.max())
    return m - y                           # absolute deviation


# --------------------------------------------------------------------------- #
#  Initial guesses
# --------------------------------------------------------------------------- #
def peak_centres(E, y, N):
    """Oscillator positions from ELF peaks, filled up with log-spaced points."""
    pk, props = find_peaks(np.log(y), prominence=0.1)
    order = np.argsort(props["prominences"])[::-1][:N]
    centres = list(E[pk[order]])
    cand = np.geomspace(E.min() * 1.5, E.max() / 1.5, 20 * N)
    while len(centres) < N:
        if not centres:
            centres.append(float(np.sqrt(E.min() * E.max())))
            continue
        lc = np.log(centres)
        dist = np.min(np.abs(np.log(cand)[:, None] - lc[None, :]), axis=1)
        centres.append(float(cand[np.argmax(dist)]))
    return np.sort(np.array(centres))


def init_amplitudes(E, y, E0, G):
    """Amplitudes are linear parameters: solve by non-negative least squares
    (relative-error weighting) for fixed E0, G."""
    Phi = basis(E, E0, G)
    A, _ = nnls(Phi / y[:, None], np.ones_like(y))
    return np.maximum(A, 1e-6)


# --------------------------------------------------------------------------- #
#  Fit
# --------------------------------------------------------------------------- #
def fit_oscillators(E, y, N, mode="log", n_starts=40, seed=1, max_nfev=3000,
                    verbose=True, floor=0.05):
    rng = np.random.default_rng(seed)
    lo = np.array([np.log(1e-9), np.log(E.min() / 2), np.log(1e-3)])
    hi = np.array([np.log(1e9), np.log(E.max() * 2), np.log(E.max() * 10)])
    lb, ub = np.tile(lo, N), np.tile(hi, N)

    best = None
    for s in range(n_starts):
        if s == 0:
            E0 = peak_centres(E, y, N)
            G = 0.3 * E0
        else:
            E0 = np.sort(np.exp(rng.uniform(np.log(E.min()), np.log(E.max()), N)))
            G = E0 * 10 ** rng.uniform(-1.3, -0.1, N)
        A = init_amplitudes(E, y, E0, G)
        theta0 = np.log(np.column_stack([A, E0, G])).ravel()
        theta0 = np.clip(theta0, lb + 1e-6, ub - 1e-6)
        try:
            sol = least_squares(residuals, theta0, bounds=(lb, ub),
                                args=(E, y, mode, floor), method="trf",
                                max_nfev=max_nfev)
        except Exception as exc:                       # pragma: no cover
            if verbose:
                print(f"  start {s:3d}: failed ({exc})")
            continue
        if verbose and (s % 10 == 0 or best is None or sol.cost < best.cost):
            print(f"  start {s:3d}: cost = {sol.cost:.5e}"
                  + ("   <-- best so far" if best is None or sol.cost < best.cost else ""))
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        sys.exit("All fit attempts failed.")

    A, E0, G = unpack(best.x)
    order = np.argsort(E0)
    return A[order], E0[order], G[order], best


# --------------------------------------------------------------------------- #
#  Sum-rule information (for reference)
# --------------------------------------------------------------------------- #
def ne_from_ep2(ep2_eV2):
    """Electron density [cm^-3] for which (hbar*omega_p)^2 = ep2 [eV^2]."""
    n_m3 = constants.epsilon_0 * constants.m_e * ep2_eV2 / constants.hbar**2
    return n_m3 * 1e-6


def sum_rule_report(E, y, fit, A, atom_density):
    """f-sum rule: int_0^inf E*ELF dE = (pi/2) * (hbar*omega_p)^2 = (pi/2)*sum(A)."""
    int_data = _trapz(E * y, E)
    int_fit = _trapz(E * fit, E)
    z_data = ne_from_ep2(2 / np.pi * int_data) / atom_density
    z_fit_rng = ne_from_ep2(2 / np.pi * int_fit) / atom_density
    z_fit_tot = ne_from_ep2(A.sum()) / atom_density
    return int_data, int_fit, z_data, z_fit_rng, z_fit_tot


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #
def write_coefficients(path, csv_path, A, E0, G, E, y, fit, mode, stats, sr):
    N = len(A)
    int_data, int_fit, z_data, z_fit_rng, z_fit_tot = sr
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Ritchie-Howie oscillator fit of the energy-loss function Im[-1/eps(E)]\n")
        f.write(f"# Source data : {csv_path}\n")
        f.write("# Model       : ELF(E) = sum_i A_i*G_i*E / ((E0_i^2 - E^2)^2 + (G_i*E)^2)\n")
        f.write(f"# Oscillators : {N}\n")
        f.write(f"# Fit range   : {E.min():.6g} - {E.max():.6g} eV  ({len(E)} points)\n")
        f.write(f"# Residual    : {mode}\n")
        f.write(f"# RMS deviation of log10(fit/data) : {stats['rms_log10']:.4f}\n")
        f.write(f"# Max |log10(fit/data)|            : {stats['max_log10']:.4f}\n")
        f.write(f"# sum(A_i) = {A.sum():.6e} eV^2  (= (hbar*omega_p)^2 if the f-sum rule were saturated)\n")
        f.write(f"# int E*ELF dE over fit range: data = {int_data:.5e}, fit = {int_fit:.5e} eV^2\n")
        f.write(f"# Effective electrons per atom: data(range) = {z_data:.3f}, "
                f"fit(range) = {z_fit_rng:.3f}, fit(total, sum A_i) = {z_fit_tot:.3f}\n")
        f.write("#\n")
        f.write(f"# {'i':>3s} {'A_i [eV^2]':>16s} {'E0_i [eV]':>16s} {'Gamma_i [eV]':>16s}\n")
        for i in range(N):
            f.write(f"  {i + 1:3d} {A[i]:16.8e} {E0[i]:16.8e} {G[i]:16.8e}\n")


def make_plot(E, y, A, E0, G, N, png_path, ylin, show):
    import matplotlib.pyplot as plt

    Eg = np.geomspace(E.min(), E.max(), 4000)
    fit_g = elf_model(Eg, A, E0, G)
    fit_d = elf_model(E, A, E0, G)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7), constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(E, y, "o", ms=3.5, mfc="none", color="tab:blue", label="data (from n, k)")
    ax1.plot(Eg, fit_g, "-", color="tab:red", lw=1.8, label=f"fit, {N} oscillators")
    for i in range(N):
        ax1.plot(Eg, A[i] * basis(Eg, E0[i:i + 1], G[i:i + 1])[:, 0],
                 "--", lw=0.8, alpha=0.55, color="gray",
                 label="individual oscillators" if i == 0 else None)
    ax1.set_xscale("log")
    if ylin:
        ax1.set_ylim(0, max(y.max(), fit_g.max()) * 1.08)
        ax1.set_ylabel(r"Im[$-1/\varepsilon$]")
    else:
        ax1.set_yscale("log")
        ax1.set_ylim(max(y.min() * 0.3, 1e-12), y.max() * 5)
        ax1.set_ylabel(r"Im[$-1/\varepsilon$]")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_title("Loss function: data vs. Ritchie-Howie fit")

    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.plot(E, fit_d / y, ".-", ms=3, lw=0.6, color="tab:red")
    ax2.set_yscale("log")
    ax2.set_ylabel("fit / data")
    ax2.set_xlabel("Energy (eV)")
    ax2.grid(True, which="both", alpha=0.25)

    fig.savefig(png_path, dpi=200)
    print(f"Plot saved to      : {png_path}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Fit the loss function (from n,k) with Ritchie-Howie oscillators.")
    ap.add_argument("csv", metavar="input", nargs="?", default="Sn_beta_Adachi_Table2.53.csv",
                    help="input file (CSV, TXT/DAT or JSON) with E [eV], n, k")
    ap.add_argument("--nk-set", type=int, default=None,
                    help="if the file has several (n, k) column pairs, use only this one "
                         "(0-based; default: per row, the first pair with valid values)")
    ap.add_argument("--format", choices=["auto", "csv", "txt", "json"], default="auto",
                    help="input format (default: auto, from the file extension)")
    ap.add_argument("-N", "--n-osc", type=int, default=None,
                    help="number of oscillators (asked interactively if omitted)")
    ap.add_argument("--emin", type=float, default=None, help="lower energy limit of fit [eV]")
    ap.add_argument("--emax", type=float, default=None, help="upper energy limit of fit [eV]")
    ap.add_argument("--mode", choices=["log", "rel", "abs", "mix"], default="log",
                    help="residual type: log (default, good over many decades), "
                         "rel (relative), abs (absolute; favours the main peaks), "
                         "mix (relative below FLOOR*max(ELF), absolute above it: "
                         "good peak shape and still follows the tails)")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="for --mode mix: ELF level (fraction of its maximum) below which "
                         "the residual becomes relative (default 0.05; smaller = more "
                         "weight on the tails, larger = closer to abs)")
    ap.add_argument("--starts", type=int, default=40, help="number of random restarts")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("--max-nfev", type=int, default=3000, help="max function evaluations per start")
    ap.add_argument("--atom-density", type=float, default=3.70e22,
                    help="atomic density [cm^-3] for the effective-electron estimate "
                         "(default: beta-Sn, 3.70e22)")
    ap.add_argument("--out-prefix", default=None,
                    help="prefix for output files (default: derived from the CSV name)")
    ap.add_argument("--ylin", action="store_true",
                    help="force a linear y axis in the main plot")
    ap.add_argument("--ylog", action="store_true",
                    help="force a logarithmic y axis in the main plot "
                         "(default: log for --mode log/rel, linear for --mode abs/mix)")
    ap.add_argument("--no-show", action="store_true", help="do not open a plot window")
    args = ap.parse_args()
    if args.ylin and args.ylog:
        ap.error("use only one of --ylin and --ylog")

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    N = args.n_osc
    if N is None:
        try:
            N = int(input("Number of oscillators: ").strip())
        except (ValueError, EOFError):
            sys.exit("Please give an integer number of oscillators.")
    if N < 1:
        sys.exit("Number of oscillators must be >= 1.")

    E, y = load_elf(args.csv, args.emin, args.emax, args.format, args.nk_set)
    print(f"Loaded {len(E)} points, E = {E.min():.4g} ... {E.max():.4g} eV, "
          f"ELF range {y.min():.3g} ... {y.max():.3g}")
    print(f"Fitting with N = {N} oscillators (residual: {args.mode}, {args.starts} starts)")

    A, E0, G, sol = fit_oscillators(E, y, N, args.mode, args.starts, args.seed, args.max_nfev,
                                    floor=args.floor)

    fit = elf_model(E, A, E0, G)
    dlog = np.log10(fit / y)
    stats = {"rms_log10": float(np.sqrt(np.mean(dlog**2))),
             "max_log10": float(np.max(np.abs(dlog)))}
    sr = sum_rule_report(E, y, fit, A, args.atom_density)

    prefix = args.out_prefix
    if prefix is None:
        prefix = os.path.splitext(args.csv)[0] + f"_RH_N{N}"
    txt_path = prefix + "_coefficients.txt"
    png_path = prefix + "_fit.png"

    write_coefficients(txt_path, args.csv, A, E0, G, E, y, fit, args.mode, stats, sr)

    print("\nOscillator coefficients (sorted by E0):")
    print(f"  {'i':>3s} {'A [eV^2]':>14s} {'E0 [eV]':>14s} {'Gamma [eV]':>14s}")
    for i in range(N):
        print(f"  {i + 1:3d} {A[i]:14.6e} {E0[i]:14.6e} {G[i]:14.6e}")
    print(f"\nRMS deviation of log10(fit/data): {stats['rms_log10']:.4f}  "
          f"(max {stats['max_log10']:.4f})")
    print(f"Effective electrons/atom: data(range) {sr[2]:.2f}, fit(range) {sr[3]:.2f}, "
          f"fit(total) {sr[4]:.2f}")
    print(f"Coefficients saved to: {txt_path}")

    if args.ylin:
        ylin = True
    elif args.ylog:
        ylin = False
    else:                                  # follow the fit mode
        ylin = args.mode in ("abs", "mix")
    make_plot(E, y, A, E0, G, N, png_path, ylin, not args.no_show)


if __name__ == "__main__":
    main()
