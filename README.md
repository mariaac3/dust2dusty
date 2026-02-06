# dust2dusty

Supernova Cosmology Analysis with MCMC - fitting intrinsic scatter distributions while accounting for selection effects using reweighting.

## Installation

### From source (development mode)

```bash
git clone https://github.com/blc56/dust2dusty.git
cd dust2dusty
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install dust2dusty
```

## Quick Start

### Command Line

```bash
# Run MCMC fitting
dust2dusty --CONFIG config/my_config.yml

# Run with debug output
dust2dusty --CONFIG config/my_config.yml --DEBUG

# Run a single likelihood evaluation (test mode)
dust2dusty --CONFIG config/my_config.yml --TEST_RUN
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
sampler = MCMC(config, pos, nwalkers, ndim, realdata, debug=True)
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
│   ├── cli.py           # Command-line interface
│   ├── dust2dust.py     # Main MCMC module
│   ├── salt2mu.py       # SALT2mu.exe interface
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
├── chains/                          # MCMC chain storage
│   ├── {data_input}-chains.h5       # Full chains (HDF5, emcee backend)
│   ├── {data_input}-autocorr.npz    # Autocorrelation time history
│   └── {data_input}-samples_thinned.npz  # Thinned samples with burn-in removed
├── logs/                            # All log files
│   ├── master.log                   # Master process: config, setup, MCMC progress
│   ├── worker_0.log                 # Worker rank 0: likelihood evaluations
│   ├── worker_1.log                 # Worker rank 1 (MPI only)
│   ├── worker_N.log                 # Worker rank N (MPI only)
│   ├── worker_salt2mu_0.log          # SALT2mu subprocess I/O for worker 0
│   ├── worker_salt2mu_1.log          # SALT2mu subprocess I/O for worker 1
│   └── worker_salt2mu_N.log          # SALT2mu subprocess I/O for worker N
├── figures/                         # Diagnostic plots
├── realdata_files/                  # Real data SALT2mu outputs
└── worker_files/                    # Per-worker SALT2mu subprocess files
    ├── {rank}_SUBPROCESS_SIM_OUT.DAT
    ├── {rank}_PYTHONCROSSTALK_OUT.DAT
    └── {rank}_SUBPROCESS_LOG_SIM.STDOUT
```

In serial mode you get `master.log` + `worker_0.log` + `worker_salt2mu_0.log`.
In MPI mode with N ranks you get `master.log` + `worker_{0..N-1}.log` + `worker_salt2mu_{0..N-1}.log`.

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

## Debugging Tools

For debugging the SALT2mu subprocess communication:

```bash
# Run SALT2mu once on real data
source RUNTEST_SUBPROCESS_BS20DATA

# Run interactive SALT2mu job
source RUNTEST_SUBPROCESS_SIM
```

## Dependencies

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- emcee >= 3.1.0
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## License

MIT License
