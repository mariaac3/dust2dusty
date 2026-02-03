"""
Utility functions for DUST2DUST that do not depend on global state.

These are pure functions that only use their input parameters,
making them easily testable and reusable.
"""

import os

import numpy as np


def set_numpy_threads(n_threads=4):
    """Set number of threads for numpy operations.

    Must be called BEFORE importing numpy to have effect.

    Args:
        n_threads: Number of threads to use (default: 4)
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)


def pconv(INP_PARAMS, paramshapesdict, splitdict, distribution_parameters):
    """
    Convert input parameters to expanded parameter list accounting for distribution shapes and splits.

    Takes high-level parameter names and expands them into full list of distribution parameters,
    accounting for:
    1. Distribution shape (Gaussian needs mu+std, Exponential needs tau, etc.)
    2. Parameter splits (e.g., different values for low/high mass, low/high redshift)

    Example:
        INP_PARAMS = ['RV']
        paramshapesdict = {'RV': 'Gaussian'}  # needs mu, std
        splitdict = {'RV': {'HOST_LOGMASS': 10}}  # split at mass=10

        Returns: ['RV_HOST_LOGMASS_low_mu', 'RV_HOST_LOGMASS_low_std',
                  'RV_HOST_LOGMASS_high_mu', 'RV_HOST_LOGMASS_high_std']

    Args:
        INP_PARAMS: List of high-level parameter names (e.g., ['c', 'RV', 'EBV'])
        paramshapesdict: Maps parameter to distribution shape (e.g., {'c': 'Gaussian'})
        splitdict: Nested dict defining parameter splits
                   {param: {split_var: split_value}}
                   e.g., {'RV': {'HOST_LOGMASS': 10, 'SIM_ZCMB': 0.1}}
        distribution_parameters: Dict mapping distribution names to their parameter names

    Returns:
        list: Expanded parameter names (length = ndim for MCMC)
              Format: 'PARAM_SPLITVAR1_lowhigh_SPLITVAR2_lowhigh_..._DISTRIBUTIONPARAM'
    """
    import itertools

    inpfull = []
    for i in INP_PARAMS:
        initial_dimension = distribution_parameters[paramshapesdict[i]]
        if i in splitdict.keys():
            things_to_split_on = splitdict[i]  # {"Mass": 10, "z": 0.1}
            nsplits = len(things_to_split_on)  # 2
            params_to_split_on = things_to_split_on.keys()  # ["Mass", "z"]
            # Create format string like "{}_{}_{}_{}" for nsplits*2 parameters
            format_string = "_".join(["{}"] * nsplits * 2)
            lowhigh_array = np.tile(
                ["low", "high"], [nsplits, 1]
            )  # [["low", "high"], ["low", "high"]]
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
    inpfull = [item for sublist in inpfull for item in sublist]
    return inpfull


def input_cleaner(
    INP_PARAMS,
    PARAMSHAPESDICT,
    SPLITDICT,
    DISTRIBUTION_PARAMETERS,
    PARAMETER_INITIALIZATION,
    parameter_overrides,
    walkfactor=2,
):
    """
    Initialize MCMC walker starting positions with appropriate constraints.

    Generates initial walker positions for emcee sampler, ensuring all parameters
    start within their valid bounds and with appropriate spreads.

    Args:
        INP_PARAMS: List of parameter names to fit (e.g., ['c', 'RV', 'EBV'])
        PARAMSHAPESDICT: Maps parameter to distribution shape
        SPLITDICT: Nested dict defining parameter splits
        DISTRIBUTION_PARAMETERS: Dict mapping distribution names to their parameter names
        PARAMETER_INITIALIZATION: Dictionary containing initialization info for each expanded parameter:
                   {param_name: [mean, std, require_positive, [lower_bound, upper_bound]]}
        parameter_overrides: Dictionary of parameters to fix (not fit)
        walkfactor: Multiplier for number of walkers (nwalkers = ndim * walkfactor, default: 2)

    Returns:
        tuple: (pos, nwalkers, ndim)
               pos: array of shape (nwalkers, ndim) with initial walker positions
               nwalkers: number of MCMC walkers
               ndim: number of dimensions (parameters)
    """
    plist = pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT, DISTRIBUTION_PARAMETERS)
    nwalkers = len(plist) * walkfactor
    for element in parameter_overrides.keys():
        plist.remove(element)
    pos = np.abs(0.1 * np.random.randn(nwalkers, len(plist)))
    for entry in range(len(plist)):
        newpos_param = PARAMETER_INITIALIZATION[plist[entry]]
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


def subprocess_to_snana(OUTDIR, snana_mapping):
    """
    Convert GENPDF file from SUBPROCESS format to SNANA-compatible format.

    Reads GENPDF.DAT file, removes the first line (header), and replaces variable
    names from subprocess format (e.g., 'SIM_c', 'SIM_RV') to SNANA format
    (e.g., 'SALT2c', 'RV') so the file can be used directly in SNANA simulations.

    Args:
        OUTDIR: Output directory containing GENPDF.DAT (should end with '/')
        snana_mapping: Dictionary mapping subprocess names to SNANA names.
                       Uses SUBPROCESS_TO_SNANA constant:
                       {'SIM_c': 'SALT2c', 'SIM_RV': 'RV', 'HOST_LOGMASS': 'LOGMASS', ...}

    Side effects:
        - Removes and recreates GENPDF.DAT file with:
          - First line removed
          - All variable names converted to SNANA format

    Returns:
        str: 'Done' upon completion
    """
    filein = OUTDIR + "GENPDF.DAT"
    f = open(filein)
    lines = f.readlines()
    f.close()
    del lines[0]
    os.remove(filein)
    f = open(filein, "w+")
    for line in lines:
        f.write(line)
    f.close()
    f = open(filein)
    filedata = f.read()
    f.close()
    for i in snana_mapping.keys():
        if i in filedata:
            filedata = filedata.replace(i, snana_mapping[i])
    os.remove(filein)
    f = open(filein, "w")
    f.write(filedata)
    f.close()
    return "Done"


def normhisttodata(datacount, simcount):
    """
    Normalize simulation histogram to match total counts in data.

    Scales simulation counts to have same total as data, computes Poisson errors,
    and masks bins where both data and sim are zero. This ensures fair comparison
    between data and simulation histograms regardless of total event counts.

    Args:
        datacount: Array of data histogram counts per bin (will be converted to numpy array)
        simcount: Array of simulation histogram counts per bin (will be converted to numpy array)

    Returns:
        tuple: (datacount_masked, simcount_normalized, poisson_errors, mask)
            datacount_masked: Data counts with zero bins removed
            simcount_normalized: Simulation counts scaled by (datatot/simtot), zeros removed
            poisson_errors: sqrt(datacount) per bin, minimum value 1 to avoid division by zero
            mask: Boolean array indicating which bins are non-zero (True = kept)
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
