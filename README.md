# dust2dusty

Supernova Cosmology Analysis with MCMC - fitting intrinsic scatter distributions while accounting for selection effects using reweighting.

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
dust2dusty --CONFIG config/my_config.yml

# Quick debug run (3 iterations only, verbose output, then exit)
dust2dusty --CONFIG config/my_config.yml --DEBUG

# Full MCMC with debug-level logging output
dust2dusty --CONFIG config/my_config.yml --DEBUG_FULL

# Run a single likelihood evaluation (test mode)
dust2dusty --CONFIG config/my_config.yml --TEST_RUN

# Use zeus or nautilus instead of emcee
dust2dusty --CONFIG config/my_config.yml --SAMPLER zeus
dust2dusty --CONFIG config/my_config.yml --SAMPLER nautilus

# Override number of walkers (emcee/zeus only)
dust2dusty --CONFIG config/my_config.yml --NWALKERS 64

# MPI run (emcee or zeus)
mpirun -n 8 dust2dusty --CONFIG config/my_config.yml --USE_MPI
```

### Python API

```python
from dust2dusty import setup_logging, get_logger, Config, load_config, init_dust2dust, MCMC

# Set up logging
setup_logging(debug=True)
logger = get_logger()

# Load configuration
config = load_config("config/my_config.yml", args)

# Initialize and run
realdata = init_dust2dust(config, debug=True)
sampler = MCMC(config, pos, nwalkers, ndim, realdata, debug=True, sampler="emcee")
```

## Configuration

A YAML configuration file is required. See `config/` directory for examples.

Required configuration keys:
- `DATA_INPUT`: Path to real data input file
- `SIM_INPUT`: Path to simulation input file
- `SIMREF_FILE`: Path to simulation reference file
- `INP_PARAMS`: List of parameters to fit (e.g., `['c', 'RV', 'EBV']`)
- `PARAMSHAPESDICT`: Distribution shapes for each parameter
- `SPLITDICT`: Parameter splits (e.g., by host mass)
- `PARAMETER_INITIALIZATION`: Prior bounds and initialization
- `SPLITARR`: Split variable arrays

## Package Structure

```
dust2dusty/
├── src/dust2dusty/
│   ├── __init__.py      # Package initialization
│   ├── cli.py           # Command-line interface and Config dataclass
│   ├── mcmc.py          # MCMC sampling (emcee, zeus, nautilus)
│   ├── dust2dust.py     # Likelihood, worker init, SALT2mu interface
│   ├── salt2mu.py       # SALT2mu.exe subprocess wrapper
│   ├── utils.py         # input_cleaner, pconv, helpers
│   └── log.py           # Shared logging configuration
├── tests/               # Test suite
├── config/              # Example configuration files
├── pyproject.toml       # Package metadata and dependencies
└── README.md
```

## Output Directory Structure

Running `dust2dusty` creates the following output tree (default `./dust2dust_output/`):

```
{outdir}/
├── chains/                               # MCMC chain storage
│   ├── {data_input}-chains.h5            # Full chains (HDF5)
│   ├── {data_input}-autocorr.npz         # Autocorrelation time history (emcee only)
│   ├── {data_input}-samples_thinned.npz  # Thinned/posterior samples
│   └── {data_input}-debug_chains.txt     # Debug run chains (--DEBUG only)
├── logs/                                 # All log files
│   ├── master.log                        # Master process: config, setup, MCMC progress
│   ├── worker_0.log                      # Worker rank 0: likelihood evaluations
│   ├── worker_N.log                      # Worker rank N (MPI only)
│   ├── worker_salt2mu_0.log              # SALT2mu subprocess I/O for worker 0
│   └── worker_salt2mu_N.log              # SALT2mu subprocess I/O for worker N
├── realdata_files/                       # Real data SALT2mu outputs
└── worker_salt2mu_files/                 # Per-worker SALT2mu subprocess files
    ├── {rank}_SUBPROCESS_SALT2MU_RES.DAT
    ├── {rank}_GENPDF_PYTHONCROSSTALK.DAT
    └── {rank}_SUBPROCESS_SALT2MU_LOG.STDOUT
```

In serial mode you get `master.log` + `worker_0.log` + `worker_salt2mu_0.log`.
In MPI mode with N ranks you get `master.log` + `worker_{0..N-1}.log` + `worker_salt2mu_{0..N-1}.log`.

## CLI Flags

### Sampler selection

| Flag | Description |
|------|-------------|
| `--SAMPLER emcee` | Ensemble MCMC sampler (default). Convergence via autocorrelation time. |
| `--SAMPLER zeus` | Ensemble slice sampler. Convergence via `AutocorrelationCallback`. |
| `--SAMPLER nautilus` | Importance sampler with neural networks. Returns Bayesian evidence `log_z`. |
| `--NWALKERS N` | Number of walkers for emcee/zeus (default: `2 * ndim`). Minimum: `2 * ndim`. |
| `--USE_MPI` | Distribute likelihood evaluations across MPI ranks. |

### Debug modes

| Flag | Logging | MCMC Behavior | SALT2mu | Use Case |
|------|---------|---------------|---------|----------|
| *(none)* | INFO (file), WARNING (console) | Full convergence run | Production (optmask=4) | Production runs |
| `--VERBOSE` | INFO (file + console) | Full convergence run | Production (optmask=4) | Monitor progress on console |
| `--DEBUG` | DEBUG (file + console) | 3 steps/calls, then exit | Debug (optmask=1, FITRES) | Quick sanity checks |
| `--DEBUG_FULL` | DEBUG (file + console) | Full convergence run | Production (optmask=4) | Diagnose issues during full runs |
| `--TEST_RUN` | DEBUG (file + console) | Single likelihood eval | Debug (optmask=1, FITRES) | Verify setup without MCMC |

- `--DEBUG`: Runs only 3 MCMC steps (emcee/zeus) or 3 likelihood calls (nautilus), saves a debug chain text file, then exits cleanly. Works for all three samplers.
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
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- zeus-mcmc (optional, for `--SAMPLER zeus`)
- nautilus-sampler (optional, for `--SAMPLER nautilus`)
- mpi4py (optional, for `--USE_MPI`)

## License

MIT License
