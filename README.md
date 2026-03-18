# dust2dusty

Supernova Cosmology Analysis with MCMC — fits intrinsic scatter distributions of supernova properties (color, stretch, dust extinction, etc.) by comparing real data to simulations reweighted via the external `SALT2mu.exe` C executable.

## Installation

### From source (development mode)

```bash
git clone https://github.com/blc56/dust2dusty.git
cd dust2dusty
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Run MCMC fitting
dust2dusty config/my_config.yml my_output_dir/

# Quick debug run (3 iterations only, DEBUG-level logging, then exit)
dust2dusty config/my_config.yml my_output_dir/ --DEBUG_RUN

# Full MCMC with DEBUG-level logging output
dust2dusty config/my_config.yml my_output_dir/ --DEBUG_FULL

# Run a single likelihood evaluation (test mode, no MCMC)
dust2dusty config/my_config.yml my_output_dir/ --TEST_RUN

# Use nautilus instead of emcee
dust2dusty config/my_config.yml my_output_dir/ --SAMPLER nautilus

# MPI run
mpirun -n 8 dust2dusty config/my_config.yml my_output_dir/

# Resume chains from a previous run
dust2dusty config/my_config.yml my_output_dir/ --RESUME old_output_dir/

# Force overwrite existing output directory
dust2dusty config/my_config.yml my_output_dir/ --FORCE_OVERRIDE

# Show INFO-level messages on the console
dust2dusty config/my_config.yml my_output_dir/ --VERBOSE
```

### Plot chains after a run

```bash
# Auto-reads parameter names from the HDF5 file
plot_chains my_output_dir/chains/data-chains.h5 -o chains.png

# Discard first 200 steps as burn-in, thin by 10
plot_chains my_output_dir/chains/data-chains.h5 -o chains.png -d 200 -t 10
```

## Configuration

A YAML configuration file is required. See `config/` directory for examples.

Required configuration keys:

- `DATA_INPUT`: Path to real data input file for SALT2mu
- `SIM_INPUT`: Path to simulation input file for SALT2mu
- `SIMREF_FILE`: Path to simulation reference file
- `FITTED_PARAMS`: Dict mapping parameter names to distribution/split config
- `PARAMETER_INITS`: Per-parameter initialization (p0, p0_std, bounds)

Optional keys:

- `SALT2MU_GENPDF_GRID`: Overrides for the default parameter grid ranges used
  when writing PDFs (merged with the built-in `_DEFAULT_PARAMETER_GRID`)

## Package Structure

```
dust2dusty/
├── src/dust2dusty/
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Config dataclass and main entry point
│   ├── mcmc.py              # MCMC sampling (emcee, nautilus)
│   ├── likelihood_worker.py # Likelihood, prior, and per-worker SALT2mu state
│   ├── salt2mu.py           # SALT2mu.exe subprocess wrapper and PDF writers
│   ├── utils.py             # Parameter expansion, initialization, histogram utilities
│   ├── plot_chains.py       # Post-run chain visualisation and summary statistics
│   └── log.py               # Shared logging configuration
├── scripts/
│   └── plot_chains.py       # Thin wrapper for the plot_chains entry point
├── tests/                   # Test suite
├── config/                  # Example configuration files
├── pyproject.toml           # Package metadata and dependencies
└── README.md
```

## Output Directory Structure

Running `dust2dusty` creates the following output tree:

```
{outdir}/
├── chains/                               # MCMC chain storage
│   ├── {data_input}-chains.h5            # Full chains (HDF5, emcee backend)
│   ├── {data_input}-autocorr.npz         # Autocorrelation time history (emcee only)
│   ├── {data_input}-samples_thinned.npz  # Thinned/posterior samples
│   └── {data_input}-debug_chains.txt     # Debug run chains (--DEBUG_RUN only)
├── logs/                                 # All log files
│   ├── master.log                        # Master process: config, setup, MCMC progress
│   ├── worker_00.log                     # Worker rank 0: likelihood evaluations
│   ├── worker_N.log                      # Worker rank N (MPI only)
│   ├── worker_salt2mu_00.log             # SALT2mu subprocess I/O for worker 0
│   └── worker_salt2mu_N.log              # SALT2mu subprocess I/O for worker N
├── realdata_salt2mu_files/               # Real data SALT2mu outputs
└── worker_salt2mu_files/                 # Per-worker SALT2mu subprocess files
    ├── {rank}_SUBPROCESS_SALT2MU_RES.DAT
    ├── {rank}_GENPDF_PYTHONCROSSTALK.DAT
    └── {rank}_SUBPROCESS_SALT2MU_LOG.STDOUT
```

In serial mode you get `master.log` + `worker_00.log` + `worker_salt2mu_00.log`.
In MPI mode with N ranks you get `master.log` + `worker_{00..N-1}.log` + `worker_salt2mu_{00..N-1}.log`.

## CLI Flags

### Sampler selection

| Flag | Description |
|------|-------------|
| `--SAMPLER emcee` | Ensemble MCMC sampler (default). Convergence via autocorrelation time. |
| `--SAMPLER nautilus` | Importance sampler with neural networks. Returns Bayesian evidence `log_z`. |
| `--FORCE_OVERRIDE` | Remove and recreate output directory if it already exists. |
| `--RESUME OLD_DIR` | Copy chains from OLD_DIR and continue sampling. |

### Debug modes

| Flag | Logging | MCMC Behavior | SALT2mu | Use Case |
|------|---------|---------------|---------|----------|
| *(none)* | INFO (file), WARNING (console) | Full convergence run | Production (optmask=4) | Production runs |
| `--VERBOSE` | INFO (file + console) | Full convergence run | Production (optmask=4) | Monitor progress on console |
| `--DEBUG_RUN` | DEBUG (file + console) | 3 steps/calls, then exit | Debug (optmask=1, FITRES) | Quick sanity checks |
| `--DEBUG_FULL` | DEBUG (file + console) | Full convergence run | Production (optmask=4) | Diagnose issues during full runs |
| `--TEST_RUN` | DEBUG (file + console) | Single likelihood eval | Debug (optmask=1, FITRES) | Verify setup without MCMC |

- `--DEBUG_RUN`: Runs only 3 MCMC steps (emcee) or 3 likelihood calls (nautilus), saves a debug chain text file, then exits cleanly.
- `--DEBUG_FULL`: Runs the full production MCMC with DEBUG-level logging to console and log files. Useful for diagnosing issues that only appear during longer runs.
- `--TEST_RUN`: Evaluates the likelihood once at the starting parameter values and exits. No MCMC sampling is performed.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/dust2dusty tests
ruff check src/dust2dusty tests
```

## Dependencies

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- emcee >= 3.1.0
- h5py
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- schwimmbad
- nautilus-sampler (optional, for `--SAMPLER nautilus`)
- mpi4py (optional, for MPI runs)

## License

MIT License
