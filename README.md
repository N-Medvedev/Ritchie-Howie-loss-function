# Ritchie-Howie loss function
Script for fitting optical coefficients with oscillators parameters in the form of Ritchie-Howie loss function

The script does the following:

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
Positional argument `input` (default: data.csv). The format is
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
    python fit_elf_ritchie_howie.py data.csv -N 8
    python fit_elf_ritchie_howie.py data.csv -N 10 --emin 1 --emax 5000 --starts 100
    python fit_elf_ritchie_howie.py data.csv            # asks for N interactively
    python fit_elf_ritchie_howie.py data.txt -N 1 --mode abs --atom-density 4.99e22
    python fit_elf_ritchie_howie.py data.json -N 4 --mode mix --floor 0.02 --nk-set 0 --no-show

Requires: numpy, pandas, scipy, matplotlib
