"""
Utility functions for DUST2DUSTY that do not depend on global state.

These are pure functions that only use their input parameters,
making them easily testable and reusable.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dust2dusty.cli import Config

from dust2dusty.salt2mu import SALT2mu

# Constants
JOBNAME_SALT2MU: str = "SALT2mu.exe"


def cmd_exe(executable: str, input_file: str) -> str:
    """
    Build command string for SALT2mu.exe with subprocess file placeholders.

    Args:
        executable: Name of the executable (e.g., 'SALT2mu.exe').
        input_file: Path to the input file for SALT2mu.

    Returns:
        Command string with %s placeholders for subprocess files
        (mapsout, SALT2muout, log).
    """
    return f"{executable} {input_file} SUBPROCESS_FILES=%s,%s,%s "


def init_salt2mu_realdata(config: Config, debug: bool = False) -> dict[str, Any]:
    """
    Initialize DUST2DUSTY by running SALT2mu on real data.

    Runs SALT2mu on real data to get baseline values for beta, betaerr,
    sigint, and siginterr that will be compared against in likelihood
    calculations. This establishes the "truth" values from observed data.

    Args:
        config: Configuration object containing paths and parameters.
        debug: If True, use debug connection index.

    Returns:
        Dictionary containing real data fit results with keys:
            - beta: Color-luminosity parameter
            - betaerr: Uncertainty on beta
            - sigint: Intrinsic scatter
            - siginterr: Uncertainty on sigint
            - bindf: Pandas DataFrame with binned statistics
            - alpha: SALT2 standardization parameter
            - alphaerr: Uncertainty on alpha
            - maxprob: Maximum probability ratio
    """
    index = ""
    directory = "realdata_files"

    if debug:
        index = "DEBUG"
    outdir = Path(config.outdir)

    subprocess_log_data = outdir / f"{directory}/{index}_SUBPROCESS_LOG_DATA.STDOUT"
    subprocess_log_data.touch()

    realdata_out = outdir / f"{directory}/SUBPROCESS_REAL_DATA_OUT.DAT"
    realdata_out.touch()

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = f"'c(6,-0.2:0.25)*{config.SPLIT_PARAMETER_FORMATS[config.splitparam]}'"

    cmd = cmd_exe(JOBNAME_SALT2MU, config.data_input) + (
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} debug_flag=930"
    )

    real_data = SALT2mu(
        cmd,
        config.outdir + "NOTHING.DAT",
        realdata_out,
        subprocess_log_data,
        is_realdata=True,
        debug=debug,
    )

    return real_data.salt2mu_results


def set_numpy_threads(n_threads: int = 4) -> None:
    """
    Set number of threads for numpy operations.

    Must be called BEFORE importing numpy to have effect. Sets environment
    variables for various BLAS implementations.

    Args:
        n_threads: Number of threads to use for linear algebra operations.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)


def pconv(
    inp_params: list[str],
    paramshapesdict: dict[str, str],
    splitdict: dict[str, dict[str, float]],
    distribution_parameters: dict[str, list[str]],
) -> list[str]:
    """
    Convert input parameters to expanded parameter list.

    Takes high-level parameter names and expands them into a full list of
    distribution parameters, accounting for:
    1. Distribution shape (Gaussian needs mu+std, Exponential needs tau, etc.)
    2. Parameter splits (e.g., different values for low/high mass)

    Example:
        >>> pconv(['RV'], {'RV': 'Gaussian'}, {'RV': {'HOST_LOGMASS': 10}},
        ...       {'Gaussian': ['mu', 'std']})
        ['RV_HOST_LOGMASS_low_mu', 'RV_HOST_LOGMASS_low_std',
         'RV_HOST_LOGMASS_high_mu', 'RV_HOST_LOGMASS_high_std']

    Args:
        inp_params: List of high-level parameter names (e.g., ['c', 'RV', 'EBV']).
        paramshapesdict: Maps parameter to distribution shape
            (e.g., {'c': 'Gaussian'}).
        splitdict: Nested dict defining parameter splits.
            Format: {param: {split_var: split_value}}.
            Example: {'RV': {'HOST_LOGMASS': 10, 'SIM_ZCMB': 0.1}}.
        distribution_parameters: Dict mapping distribution names to their
            parameter names (e.g., {'Gaussian': ['mu', 'std']}).

    Returns:
        Expanded parameter names (length = ndim for MCMC).
        Format: 'PARAM_SPLITVAR1_lowhigh_SPLITVAR2_lowhigh_..._DISTPARAM'.
    """
    inpfull: list[list[str]] = []
    for i in inp_params:
        initial_dimension = list(distribution_parameters[paramshapesdict[i]])
        if i in splitdict.keys():
            things_to_split_on = splitdict[i]
            nsplits = len(things_to_split_on)
            params_to_split_on = things_to_split_on.keys()
            # Create format string like "{}_{}_{}_{}" for nsplits*2 parameters
            format_string = "_".join(["{}"] * nsplits * 2)
            lowhigh_array = np.tile(["low", "high"], [nsplits, 1])
            splitlist = []
            for lowhigh_combo in itertools.product(*lowhigh_array):
                to_format = [val for pair in zip(params_to_split_on, lowhigh_combo) for val in pair]
                final = format_string.format(*to_format)
                splitlist.append(final)
            initial_dimension = [
                tmp[1] + "_" + tmp[0] for tmp in itertools.product(splitlist, initial_dimension)
            ]
        final_dimension = [i + "_" + s for s in initial_dimension]
        inpfull.append(final_dimension)
    return [item for sublist in inpfull for item in sublist]


def input_cleaner(
    inp_params: list[str],
    paramshapesdict: dict[str, str],
    splitdict: dict[str, dict[str, float]],
    distribution_parameters: dict[str, list[str]],
    parameter_initialization: dict[str, list[Any]],
    parameter_overrides: dict[str, float],
    walkfactor: int = 2,
) -> tuple[NDArray[np.float64], int, int]:
    """
    Initialize MCMC walker starting positions with appropriate constraints.

    Generates initial walker positions for emcee sampler, ensuring all
    parameters start within their valid bounds and with appropriate spreads.

    Args:
        inp_params: List of parameter names to fit (e.g., ['c', 'RV', 'EBV']).
        paramshapesdict: Maps parameter to distribution shape.
        splitdict: Nested dict defining parameter splits.
        distribution_parameters: Dict mapping distribution names to parameter names.
        parameter_initialization: Dictionary containing initialization info for
            each expanded parameter. Format:
            {param_name: [mean, std, require_positive, [lower_bound, upper_bound]]}.
        parameter_overrides: Dictionary of parameters to fix (not fit).
        walkfactor: Multiplier for number of walkers (nwalkers = ndim * walkfactor).

    Returns:
        Tuple of (pos, nwalkers, ndim) where:
            - pos: Array of shape (nwalkers, ndim) with initial walker positions
            - nwalkers: Number of MCMC walkers
            - ndim: Number of dimensions (parameters)
    """
    plist = pconv(inp_params, paramshapesdict, splitdict, distribution_parameters)
    nwalkers = len(plist) * walkfactor
    for element in parameter_overrides.keys():
        plist.remove(element)
    pos = np.abs(0.1 * np.random.randn(nwalkers, len(plist)))
    for entry in range(len(plist)):
        newpos_param = parameter_initialization[plist[entry]]
        pos[:, entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:, entry]))
        if newpos_param[2]:
            pos[:, entry] = np.abs(pos[:, entry])
        while any(ele < newpos_param[3][0] for ele in pos[:, entry]) or any(
            ele > newpos_param[3][1] for ele in pos[:, entry]
        ):
            pos[:, entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:, entry]))
            if newpos_param[2]:
                pos[:, entry] = np.abs(pos[:, entry])
    return pos, nwalkers, len(plist)


def subprocess_to_snana(outdir: str, snana_mapping: dict[str, str]) -> str:
    """
    Convert GENPDF file from SUBPROCESS format to SNANA-compatible format.

    Reads GENPDF.DAT file, removes the first line (header), and replaces
    variable names from subprocess format (e.g., 'SIM_c', 'SIM_RV') to SNANA
    format (e.g., 'SALT2c', 'RV') so the file can be used directly in SNANA
    simulations.

    Args:
        outdir: Output directory containing GENPDF.DAT (should end with '/').
        snana_mapping: Dictionary mapping subprocess names to SNANA names.
            Example: {'SIM_c': 'SALT2c', 'SIM_RV': 'RV', 'HOST_LOGMASS': 'LOGMASS'}.

    Returns:
        'Done' upon successful completion.

    Side Effects:
        Modifies GENPDF.DAT file in place:
        - Removes first line
        - Converts all variable names to SNANA format
    """
    filein = outdir + "GENPDF.DAT"
    with open(filein) as f:
        lines = f.readlines()
    del lines[0]
    os.remove(filein)
    with open(filein, "w+") as f:
        for line in lines:
            f.write(line)
    with open(filein) as f:
        filedata = f.read()
    for key in snana_mapping.keys():
        if key in filedata:
            filedata = filedata.replace(key, snana_mapping[key])
    os.remove(filein)
    with open(filein, "w") as f:
        f.write(filedata)
    return "Done"


def normhisttodata(
    datacount: NDArray[np.float64] | list[float],
    simcount: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """
    Normalize simulation histogram to match total counts in data.

    Scales simulation counts to have same total as data, computes Poisson
    errors, and masks bins where both data and sim are zero. This ensures
    fair comparison between data and simulation histograms regardless of
    total event counts.

    Args:
        datacount: Array of data histogram counts per bin.
        simcount: Array of simulation histogram counts per bin.

    Returns:
        Tuple of (datacount_masked, simcount_normalized, poisson_errors, mask):
            - datacount_masked: Data counts with zero bins removed
            - simcount_normalized: Sim counts scaled by (datatot/simtot), zeros removed
            - poisson_errors: sqrt(datacount) per bin, minimum value 1
            - mask: Boolean array indicating non-zero bins (True = kept)
    """
    datacount = np.array(datacount)
    simcount = np.array(simcount)
    datatot = np.sum(datacount)
    simtot = np.sum(simcount)
    simcount = simcount * datatot / simtot

    ww = (datacount != 0) | (simcount != 0)

    poisson = np.sqrt(datacount)
    poisson[datacount == 0] = 1
    poisson[~np.isfinite(poisson)] = 1
    return datacount[ww], simcount[ww], poisson[ww], ww
