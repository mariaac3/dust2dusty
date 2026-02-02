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
dust2dusty --CONFIG config/my_config.yml --test_run
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
│   └── logging.py       # Shared logging configuration
├── tests/               # Test suite
├── config/              # Example configuration files
├── pyproject.toml       # Package metadata and dependencies
└── README.md
```

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
