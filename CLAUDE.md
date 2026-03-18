# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dust2dusty** is a Python package for supernova cosmology MCMC analysis. It fits intrinsic scatter distributions of supernova properties (color, stretch, dust extinction, etc.) by comparing real data to simulations reweighted via the external `SALT2mu.exe` C executable.

## Installation & Setup

```bash
pip install -e ".[dev]"   # Editable install with dev tools
```

**Entry point:**
```bash
dust2dusty --CONFIG config.yml [--VERBOSE] [--DEBUG] [--DEBUG_FULL] [--TEST_RUN] [--USE_MPI] [--SAMPLER emcee|zeus|nautilus]
```

## Development Commands

```bash
# Run tests
pytest

# Run a single test file
pytest tests/test_logging.py

# Linting
ruff check dust2dusty/

# Formatting
black dust2dusty/

# Type checking
mypy dust2dusty/
```

## Debug Modes

| Flag | MCMC iterations | SALT2mu mode | Purpose |
|------|-----------------|--------------|---------|
| (none) | Full convergence | production | Full analysis |
| `--VERBOSE` | Full convergence | production | Monitor progress |
| `--DEBUG` | 3 only | FITRES | Quick sanity check |
| `--DEBUG_FULL` | Full convergence | production | Diagnose long-run issues |
| `--TEST_RUN` | 1 likelihood eval | FITRES | Verify setup |

## Architecture

### Data Flow

```
CLI (cli.py: main)
  → Load YAML config → Config dataclass
  → Run SALT2mu on real data (utils.py: init_salt2mu_realdata)
  → MCMC sampling loop (mcmc.py: MCMC)
      → log_probability (likelihood_worker.py)
          → Write PDF functions → SALT2mu subprocess (salt2mu.py)
          → Parse SALT2mu output → chi-squared likelihood
  → Save chains (HDF5 + thinned samples)
  → Shutdown all SALT2mu subprocesses
```

### Module Responsibilities

- **`cli.py`** — `Config` dataclass (YAML loading, validation), output directory creation, `SUBPROCESS_TO_SNANA` mapping, `main()` entry point. `fitted_params` is a **dict** (not a list) mapping parameter names to their config dicts.
- **`likelihood_worker.py`** — `log_probability()`, `log_likelihood()`, `log_prior()`. Worker-local globals (`_WORKER_*`) store per-process SALT2mu state. Observables: color (c), stretch (x1), Hubble residuals (MURES) and RMS by mass bin, beta, sigint. `log_probability` converts the raw theta array to `theta_dic` (applying de-logging) before calling `log_prior` and `log_likelihood`, both of which receive a `dict[str, NDArray]`.
- **`mcmc.py`** — `MCMC()`: supports `emcee` (default), `zeus`, and `nautilus` samplers, selected via `--SAMPLER`. emcee uses an HDF5 backend with manual convergence monitoring (chain > 100×tau AND tau change < 1%); zeus uses `AutocorrelationCallback` + `SaveProgressCallback`; nautilus is an importance sampler with neural networks that builds a uniform `Prior` from `parameter_initialization` bounds, calls `log_likelihood` directly (prior handled internally), and saves `log_z` (Bayesian evidence) alongside posterior samples. emcee/zeus share schwimmbad pool/MPI setup; nautilus bypasses schwimmbad and uses `MPIPoolExecutor` directly (or `pool=None` for serial). Internal helpers `_run_emcee` / `_run_zeus` / `_run_nautilus` / `_finalize_*` contain sampler-specific logic.
- **`salt2mu.py`** — `SALT2mu` class: launches persistent `SALT2mu.exe` subprocess, communicates via files (`pdf_crosstalk_file` written by Python, `salt2mu_out` written by subprocess). Holds class-level `PARAM_TO_SALT2MU` mapping (moved here from `cli.py`) and `DEFAULT_PARAMETER_GRID` for PDF evaluation arrays. PDF writing uses the unified `write_GENPDF` method supporting arbitrary N-dimensional splits (replaces old `write1Dprobs`/`write2Dprobs`/`write3Dprobs`). `_OPERATOR_MAP` drives split boundary logic via `operator.lt`/`operator.gt`. Distribution helpers (`get_1d_asym_gauss`, `get_1d_exponential`, `get_1d_lognormal`) are `@staticmethod` and return only the probability array.
- **`utils.py`** — `pconv()` parameter expansion, `input_cleaner()` walker initialization, histogram normalization, `cmd_salt2mu_exe()` for building SALT2mu command strings.
- **`log.py`** — `setup_logging()`, `get_logger()`, `setup_salt2mu_logger()` for per-worker logging.

### SALT2mu Subprocess Communication

All SALT2mu communication is **file-based** (no pipes/sockets):
- Python writes PDF functions to `pdf_crosstalk_file`
- `SALT2mu.exe` reweights simulation and writes results to `salt2mu_out`
- Each MPI worker has its own subprocess and file pair

### Parameter System

Parameters are specified at a high level in the YAML config and expanded by `pconv()` (in `utils.py`) via two dimensions:
1. **Distribution shape**: `Gaussian` → `[mu, std]`, `Exponential` → `[Tau]`
2. **Splits**: Different values per `HOST_LOGMASS` or `SIM_ZCMB` bins

Example: `RV` with Gaussian shape and `HOST_LOGMASS` split at 10 expands to `RV_mu_HOST_LOGMASS_low`, `RV_mu_HOST_LOGMASS_high`, `RV_std_HOST_LOGMASS_low`, `RV_std_HOST_LOGMASS_high`.

### MPI Execution

- Master (rank 0): runs the selected sampler
- emcee/zeus workers (rank > 0): evaluate likelihoods via schwimmbad MPIPool; each maintains its own SALT2mu subprocess
- nautilus workers: managed internally by `mpi4py.futures.MPIPoolExecutor(initializer=_init_worker, initargs=...)`; each worker has its SALT2mu subprocess ready before servicing any likelihood calls
- Exception hook aborts all ranks on failure

## Output Structure

```
./dust2dust_output/
├── chains/          # HDF5 chains, autocorr history, thinned samples
├── logs/            # master.log, worker_N.log, worker_salt2mu_N.log
├── realdata_files/  # Real data SALT2mu output and PDF file
└── worker_salt2mu_files/  # Per-worker SALT2mu I/O files
```

## Configuration

See `config/` for example YAML files. Key sections:
- `DATA_INPUT`, `SIM_INPUT`, `SIMREF_FILE`, `OUTDIR` — file paths
- `FITTED_PARAMS`, `PARAM_DISTS`, `SPLITPARAM`, `SPLITDICT` — what to fit and how to split
- `PARAMS` — initial parameter values (before expansion)
- `SPLITARR` — arrays for evaluating PDFs at split boundaries
- `PARAMETER_INITIALIZATION` — per-parameter start, stdev, bounds, `require_positive`
- `salt2mu_genpdf_grid` — optional overrides for the default parameter grid ranges used when writing PDFs (merged with `SALT2mu.DEFAULT_PARAMETER_GRID`)

## Key Files

- `dust2dusty/likelihood_worker.py` — likelihood evaluation (core science logic)
- `dust2dusty/mcmc.py` — MCMC sampling and convergence
- `dust2dusty/salt2mu.py` — SALT2mu.exe subprocess wrapper
- `dust2dusty/cli.py` — Config dataclass and entry point
- `dust2dusty/utils.py` — parameter expansion (`pconv`) and utilities
- `config/config_popovic2023.yml` — most complete example configuration
- `legacy/` — old monolithic scripts kept for reference only
