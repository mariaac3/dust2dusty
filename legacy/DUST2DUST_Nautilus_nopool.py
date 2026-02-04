#!/usr/bin/env python
import sys
from pathlib import Path

import callSALT2mu
import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

import corner
import pylab as plt
from scipy.stats import truncnorm

# from multiprocessing import Pool
# from multiprocessing import current_process
# from multiprocessing import cpu_count
current_process = 1
# import emcee
import argparse
import itertools
import logging

import nautilus
import yaml

# from schwimmbad import MPIPool
JOBNAME_SALT2mu = "SALT2mu.exe"  # public default code

# this prevents further parallelisation of code called from here
# SNANA or functions in numpy, scipy may do this (check with Rick)

# os.environ["OMP_NUM_THREADS"] = "1"

ncbins = 6

# Nautilus settings
# samples will be saved at the end of the run as points, log_w, log_z to CHAINS as specified in CONFIG file

npool = 1  # match this to nodes * cpus * tasks from your slurm job?

resume = False
checkpoint = "nautilus_test.hdf5"  # the hdf5 file allows runs to be resumed, but does not allow samples to be easily accessed
n_eff = 1000  # generally, n_eff = 10000 will be best. 1000 could be used for a quick test. The error on log Z will be ~ 1/sqrt(n_eff)
n_like_max = 10000  # change this to 1 if you just want to retrieve and save results from previous .hdf5 checkpoint
seed = 5768
discard_exploration = (
    False  # a bit like removing burnin if True (but probably shouldn't be necessary)
)

# work on array_conv

# ===========================================================================================================================================
############################################################# IO ###################################################
# ===========================================================================================================================================


def prep_config(args, config):
    DATA_INPUT = config["DATA_INPUT"]
    SIM_INPUT = config["SIM_INPUT"]
    INP_PARAMS = config["INP_PARAMS"]
    OUTDIR = ""
    CHAINS = config["CHAINS"]
    PARAMS = config["PARAMS"]
    PARAMSHAPESDICT = config["PARAMSHAPESDICT"]
    SPLITDICT = config["SPLITDICT"]
    CLEANDICT = config["CLEANDICT"]
    SPLITARR = config["SPLITARR"]
    SPLITPARAM = config["SPLITPARAM"]
    SIMREF_FILE = config["SIMREF_FILE"]

    SINGLE = args.SINGLE
    DEBUG = args.DEBUG
    if SINGLE:
        DEBUG = True
    NOWEIGHT = args.NOWEIGHT
    DOPLOT = args.DOPLOT
    CMD_DATA = args.CMD_DATA
    CMD_SIM = args.CMD_SIM
    GENPDF_ONLY = args.GENPDF_ONLY

    if config["OUTDIR"]:
        OUTDIR = config["OUTDIR"]
        if not OUTDIR.endswith("/"):
            OUTDIR += "/"
        print(f"Using custom directory {OUTDIR}!")
        if os.path.exists(OUTDIR):
            print(f"{OUTDIR} already exists! I will not remake it then!")
        else:
            os.mkdir(OUTDIR)
        subdir_list = ["chains", "figures", "parallel", "logs"]
        for subdir in subdir_list:
            os.mkdir(OUTDIR + subdir)
        if not (
            (os.path.isdir(OUTDIR + "chains"))
            and (os.path.isdir(OUTDIR + "figures"))
            and (os.path.isdir(OUTDIR + "parallel"))
        ):
            print("Please make sure that the following folders exist in your new directory:")
            print("chains, figures, parallel")
            print("One or more of these does not exist. Quitting gracefully.")
            quit()
    else:
        OUTDIR = os.getcwd() + "/"

    print("Done assigning variables.")
    return (
        DATA_INPUT,
        INP_PARAMS,
        OUTDIR,
        SINGLE,
        DEBUG,
        NOWEIGHT,
        DOPLOT,
        CHAINS,
        CMD_DATA,
        CMD_SIM,
        PARAMS,
        SIM_INPUT,
        PARAMSHAPESDICT,
        SPLITDICT,
        CLEANDICT,
        SPLITARR,
        SIMREF_FILE,
        GENPDF_ONLY,
        SPLITPARAM,
    )
    # END prep_connection


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("seaborn").setLevel(logging.ERROR)
    # END setup_logging


def load_config(config_path):
    with open(config_path) as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.FullLoader)
    return config
    # END load_config


def get_args():
    parser = argparse.ArgumentParser()

    msg = "HELP menu for config options"

    msg = "Directory for configuration file."
    parser.add_argument("--CONFIG", help=msg, type=str, default="")

    msg = "Does not launch the MCMC, and instead runs a SINGLE subprocess on the proposed set of parameters. Use for testing small changes."
    parser.add_argument("--SINGLE", help=msg, type=bool, default=False)

    msg = "General purpose DEBUG flag. Prints more information."
    parser.add_argument("--DEBUG", help=msg, type=bool, default=False)

    msg = "Disables weighting function temporarily. Use when there is a sim with pre-existing distribution that needs to be compared with data without reweighting (EG G10, C11.)"
    parser.add_argument("--NOWEIGHT", help=msg, type=bool, default=False)

    msg = "Creates corner and chains plots without running the full MCMC. Requires chains to be specified."
    parser.add_argument("--DOPLOT", help=msg, type=bool, default=False)

    msg = "Command line override to be applied to the SALT2mu input file for the data."
    parser.add_argument("--CMD_DATA", help=msg, type=str, default=None)

    msg = "Command line override to be applied to the SALT2mu input file for the simulation."
    parser.add_argument("--CMD_SIM", help=msg, type=str, default=None)

    msg = "If True, will create a simulation-appropriate GENPDF file before quitting. Does not run the full DUST2DUST."
    parser.add_argument("--GENPDF_ONLY", help=msg, type=bool, default=False)

    # parse it
    args = parser.parse_args()
    return args
    # END get_args


# ===================================================
###################### Dictionaries and Definitions
# ===================================================
# paramdict is hard coded to take the input parameters and expand into the necessary variables to properly model those

"""
paramdict = {'c':['c_m', 'c_std'], #Default colour, shaped as a Gaussian
             'x1':['x1_m', 'x1_l', 'x1_r'], #Default stretch, an aysmmetric (skewed) Gaussian
             'EBV':['EBV_Tau_low','EBV_Tau_high'], #Default EBV split on mass, an exponential distribution
             'RV':['RV_m_low','RV_std_low', 'RV_m_high','RV_std_high'], #Default RV is split on Mass, Gaussian
             'beta':['beta_m', 'beta_std'], #Default beta, Gaussian
             'EBVZ':['EBVZL_Tau_low','EBVZL_Tau_high', 'EBVZH_Tau_low','EBVZH_Tau_high']} #EBV split on mass, z, Exponential
"""

# paramdict will be replaced with shapedict, which defines the parameters for a given distribution. Assigning shapes to a fitted parameter will be done in the yaml input.

shapedict = {
    "Gaussian": ["mu", "std"],
    "Skewed Gaussian": ["mu", "std_l", "std_r"],
    "Exponential": ["Tau"],
    "LogNormal": ["ln_mu", "ln_std"],
    "Double Gaussian": ["a1", "mu1", "std1", "mu2", "std2"],
}


# TESTING - paramshapesdict will be part of the yaml thingy
# paramshapesdict = {'c': 'Gaussian', 'RV':'Gaussian', 'EBV':'Exponential', 'beta':'Gaussian'}
# splitdict = {'RV': {'Mass': 10 }, 'EBV': {'Mass': 10, 'z': 0.1}}
# TESTING

# cleandict is ironically named at this point as it's gotten more and more unwieldy. It is designed to contain the following:
# first entry is starting mean value for walkers. Second is the walker std. Third is whether or not the value needs to be positive (eg stds). Fourth is a list containing the lower and upper valid bounds for that parameter.


simdict = {
    "c": "SIM_c",
    "x1": "SIM_x1",
    "HOST_LOGMASS": "HOST_LOGMASS",
    "Mass": "HOST_LOGMASS",
    "RV": "SIM_RV",
    "EBV": "SIM_EBV",
    "beta": "SIM_beta",
    "SIM_ZCMB": "SIM_ZCMB",
    "EBVZ": "SIM_EBV",
    "ZTRUE": "SIM_ZCMB",
    "z": "SIM_ZCMB",
    "HOST_COLOR": "HOST_COLOR",
}  # converts inp_param into SALT2mu readable format
snanadict = {
    "SIM_c": "SALT2c",
    "SIM_RV": "RV",
    "HOST_LOGMASS": "LOGMASS",
    "SIM_EBV": "EBV",
    "SIM_ZCMB": "ZTRUE",
    "SIM_beta": "SALT2BETA",
    "HOST_COLOR": "COLOR",
}
arrdict = {
    "c": np.arange(-0.5, 0.5, 0.01),
    "x1": np.arange(-5, 5, 0.1),
    "RV": np.arange(0, 8, 0.1),
    "EBV": np.arange(0.0, 1.5, 0.02),
    "EBVZ": np.arange(0.0, 1.5, 0.02),
}  # arrays.

splitparamdict = {
    "HOST_LOGMASS": "HOST_LOGMASS(2,0:20)",
    "HOST_COLOR": "HOST_COLOR(2,-.5:2.5)",
    "zHD": "zHD(2,0:1)",
}

override = {}
# =======================================================
################### FUNCTIONS ##########################
# =======================================================


def thetaconverter(
    theta,
):  # takes in theta and returns a dictionary of what cuts to make when reading/writing theta
    thetadict = {}
    extparams = pconv(
        INP_PARAMS, PARAMSHAPESDICT, SPLITDICT
    )  # expanded list of all variables. len is ndim.
    for p in INP_PARAMS:
        thetalist = []
        for n, ep in enumerate(extparams):
            if p in ep:  # for instance, if 'c' is in 'c_l', then this records that position.
                thetalist.append(n)
        thetadict[p] = thetalist
    return thetadict  # key gives location of relevant parameters in extparams
    # END thetaconverter


def thetawriter(
    theta, key, names=False
):  # this does the splitting that thetaconverter sets up. Used in log_likelihood
    thetadict = thetaconverter(theta)
    lowbound = thetadict[key][0]
    highbound = thetadict[key][-1] + 1
    if names:
        return names[lowbound:highbound]
    else:
        return theta[
            lowbound:highbound
        ]  # Returns theta in the range of first to last index for relevant parameter. For example, inp_param = ['c', 'RV'], thetawriter(theta, 'c') would give theta[0:2] which is ['c_m', 'c_std']


# def input_cleaner(INP_PARAMS, CLEANDICT, override, walkfactor=2):  #this function takes in the input parameters and generates the walkers with appropriate dimensions, starting points, walkers, and step size
#     plist = pconv(INP_PARAMS,PARAMSHAPESDICT, SPLITDICT)
#     for element in override.keys():
#         plist.remove(element)
#     pos = np.abs(0.1 * np.random.randn(len(plist)*walkfactor, len(plist)))
#     for entry in range(len(plist)):
#         newpos_param = CLEANDICT[plist[entry]]
#         pos[:,entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:,entry]))
#         if newpos_param[2]:
#             pos[:,entry] = np.abs(pos[:,entry])
#         while ( any(ele < newpos_param[3][0] for ele in pos[:,entry]) or any(ele > newpos_param[3][1] for ele in pos[:,entry])):
#             pos[:,entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:,entry]))
#             if newpos_param[2]:
#                 pos[:,entry] = np.abs(pos[:,entry])
#     return pos, len(plist)*walkfactor, len(plist)
#     #END input_cleaner


def pconv(
    INP_PARAMS, paramshapesdict, splitdict
):  # takes paramshapedict and splitdict (both are yaml inputs) and compares them to shapedict
    """
    This takes the INP_PARAMS and expands them to be the appropriate length based on desired splits.
    For instance, if RV is defined as a Gaussian and you want to split on mass, this should create an array of length four:
    [mu_RV_low_mass, std_RV_low_mass, mu_RV_high_mass, std_RV_high_mass]
    """
    inpfull = []
    for i in INP_PARAMS:
        initial_dimension = shapedict[paramshapesdict[i]]
        if i in splitdict.keys():
            things_to_split_on = splitdict[i]  # {"Mass": 10, "z": 0.1}
            nsplits = len(things_to_split_on)  # 2
            params_to_split_on = things_to_split_on.keys()  # ["Mass", "z"]
            format_string = "{}_" * nsplits * 2  # "{}_{}_{}_{}_"
            format_string = format_string[:-1]  # remove extra "_" at the end "{}_{}_{}_{}"
            lowhigh_array = np.tile(
                ["low", "high"], [nsplits, 1]
            )  # [["low", "high"], ["low", "high"]]
            splitlist = []
            for lowhigh_combo in itertools.product(*lowhigh_array):
                to_format = [val for pair in zip(params_to_split_on, lowhigh_combo) for val in pair]
                final = format_string.format(*to_format)
                splitlist.append(final)
            # initial_dimension = [tmp[0]+'_'+tmp[1] for tmp in itertools.product(initial_dimension,splitlist)]
            initial_dimension = [
                tmp[1] + "_" + tmp[0] for tmp in itertools.product(splitlist, initial_dimension)
            ]
        final_dimension = [i + "_" + s for s in initial_dimension]
        inpfull.append(final_dimension)
    inpfull = [item for sublist in inpfull for item in sublist]
    return inpfull
    # END split_cleaner


def array_conv(inp, SPLITDICT, SPLITARR):
    if (inp == "beta") or (inp == "alpha"):
        return []
    arrlist = []
    arrlist.append(arrdict[inp])
    if inp in SPLITDICT.keys():
        for s in SPLITDICT[inp].keys():
            arrlist.append(eval(SPLITARR[s]))
    return arrlist
    # END array_conv


def xconv(inp_param):  # gnarly but needed for plotting
    if inp_param == "c":
        return np.linspace(-0.3, 0.3, 12)
    elif inp_param == "x1":
        return np.linspace(-3, 3, 12)
    # END xconv


def dffixer(df, RET, ifdata):
    cpops = []
    rmspops = []

    dflow = df.loc[df[f"ibin_{SPLITPARAM}"] == 0]
    dfhigh = df.loc[df[f"ibin_{SPLITPARAM}"] == 1]

    lowNEVT = dflow.NEVT.values
    highNEVT = dfhigh.NEVT.values
    lowrespops = dflow.MURES_SUM.values
    highrespops = dfhigh.MURES_SUM.values

    for q in np.unique(df.ibin_c.values):
        cpops.append(np.sum(df.loc[df.ibin_c == q].NEVT))

    cpops = np.array(cpops)
    lowRMS = dflow.STD_ROBUST.values
    highRMS = dfhigh.STD_ROBUST.values

    if RET == "HIST":
        return cpops
    elif RET == "ANALYSIS":
        return (
            (cpops),
            (highrespops / dfhigh.NEVT.values),
            (lowrespops / dflow.NEVT.values),
            (highRMS),
            (lowRMS),
            (highNEVT),
            (lowNEVT),
        )
    else:
        return "No output"
    # END dffixer


def LL_Creator(
    inparr, simbeta, simsigint, returnall_2=False
):  # takes a list of arrays - eg [[data_1, sim_1],[data_2, sim_2]] and gives an LL
    RMS_weight = 1
    if returnall_2:
        datacount_list = []
        simcount_list = []
        poisson_list = []
    LL_list = []
    print("real beta, sim beta, real beta error", realbeta, simbeta, realbetaerr, flush=True)
    LL_Beta = -0.5 * ((realbeta - simbeta) ** 2 / realbetaerr**2)  # should use realbetaerr here
    LL_sigint = -0.5 * ((realsigint - simsigint) ** 2 / realsiginterr**2)  # ditto
    # thetaconverter(INP_PARAMS)
    for n, i in enumerate(inparr):
        if n == 0:  # colour
            datacount, simcount, poisson, ww = normhisttodata(i[0], i[1])
        elif n == 1:  # Hi mass MURES
            datacount = i[0]
            simcount = i[1]
            poisson = inparr[3][0] / np.sqrt(inparr[-2][0])
        elif n == 2:  # lo mass MURES
            datacount = i[0]
            simcount = i[1]
            poisson = inparr[4][0] / np.sqrt(inparr[-1][0])
        elif n == 3:  # Hi mass RMS
            datacount = i[0]
            simcount = i[1]
            poisson = i[0] / np.sqrt(2 * inparr[-2][0])
        elif n == 4:  # low mass RMS
            datacount = i[0]
            simcount = i[1]
            poisson = i[0] / np.sqrt(2 * inparr[-1][0])
        LL_c = -0.5 * np.sum((datacount - simcount) ** 2 / poisson**2)
        if n == 3:
            LL_c = LL_c * RMS_weight
        elif n == 4:
            LL_c = LL_c * RMS_weight
        LL_list.append(LL_c)
        if returnall_2:
            datacount_list.append(datacount)
            simcount_list.append(simcount)
            poisson_list.append(poisson)
    # print('A gentle reminder that we are not weighting by RMS at present.')
    LL_list = LL_list[:-2]
    LL_list.append(LL_Beta)
    LL_list.append(LL_sigint)
    LL_list = np.array(LL_list)
    print(
        "I got a LogLike breakdown of ",
        LL_list,
        " (colour, himass res, lomass res, himass std, lomass std, beta, sigint)",
        flush=True,
    )
    if not returnall_2:
        return np.nansum(LL_list)
    else:
        return (LL_list), datacount_list, simcount_list, poisson_list
    # END LL_Creator


def pltting_func(samples, INP_PARAMS, ndim):
    labels = pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT)
    for k in override.keys():
        labels.remove(k)
    plt.clf()
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    for it in range(ndim):
        ax = axes[it]
        ax.plot(samples[:, :, it], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        # ax.set_ylabel(r'$'+str(labels[it])+'$')
        ax.set_ylabel(it)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(
        OUTDIR + "figures/" + DATA_INPUT.split(".")[0].split("/")[-1] + "-chains.pdf",
        bbox_inches="tight",
    )
    print("upload " + OUTDIR + "figures/chains.pdf")
    plt.close()

    flat_samples = samples.reshape(-1, samples.shape[-1])

    plt.clf()
    fig = corner.corner(flat_samples, labels=labels, smooth=True)
    plt.savefig(OUTDIR + "figures/" + DATA_INPUT.split(".")[0].split("/")[-1] + "-corner.pdf")
    print("upload " + OUTDIR + "figures/corner.pdf")
    plt.close()
    # END pltting_func


def Criteria_Plotter(theta, genpdf_only=False):
    tc = init_connection(299, real=False, debug=True)[1]
    try:
        chisq, datacount_list, simcount_list, poisson_list = log_likelihood(
            (theta), returnall=True, connection=tc, genpdf_only=genpdf_only
        )
    except TypeError:
        if genpdf_only:
            # Read in and change the GENPDF filenames to be appropriate for SNANA-usage. The SUBPROCESS and SIM parameters are not the same by default.
            print(
                "If you did not mean to generate the GENPDF only, something has gone wrong. Otherwise this is working properly."
            )
            os.rename("parallel/299_PYTHONCROSSTALK_OUT.DAT", "GENPDF.DAT")
            return ()
        else:
            print(
                "LL was not returned after running log_likelihood, which is likely due to bad parameters. Will skip plotting."
            )
            return
    cbins = np.linspace(-0.2, 0.25, ncbins)
    chisq = -2 * chisq
    if DEBUG:
        print("RESULT!", chisq, flush=True)
    sys.stdout.flush()
    plt.rcParams.update({"text.usetex": True, "font.size": 12})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ##### Colour Hist
    ax = axs[0]
    ax.errorbar(
        cbins, datacount_list[0], yerr=(poisson_list[0]), fmt="o", c="darkmagenta", label="Data"
    )
    ax.plot(cbins, simcount_list[0], c="dimgray", label="Simulation")
    ax.legend()
    ax.set_xlabel(r"$c$")
    ax.set_ylabel("Count")
    thestring = r"$\chi^2_c =$ " + str(np.around(chisq[0], 1))
    ax.text(
        -0.2,
        50,
        thestring,
    )
    ax.text(-0.2, 450, "a)")
    ###### MURES hi and lo
    ax = axs[1]
    ax.errorbar(
        cbins, datacount_list[1], yerr=(poisson_list[1]), fmt="^", c="k", label="Data, High"
    )
    ax.plot(cbins, simcount_list[1], c="tab:orange", label="Simulation, High", ls="--")
    ax.errorbar(
        cbins, datacount_list[2], yerr=(poisson_list[2]), fmt="s", c="tab:green", label="Data, Low"
    )
    ax.plot(cbins, simcount_list[2], c="tab:blue", label="Simulation, Low")
    ax.legend(bbox_to_anchor=[1.7, 1.2], ncol=2)
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$\mu - \mu_{\rm model}$")
    thestring = r"High $\chi^2_{\mu_{\rm res}} =$ " + str(np.around(chisq[1], 1))
    ax.text(
        -0.2,
        0.205,
        thestring,
    )
    thestring = r"Low $\chi^2_{\mu_{\rm res}} =$ " + str(np.around(chisq[2], 1))
    ax.text(
        -0.2,
        0.18,
        thestring,
    )
    ax.text(-0.2, 0.275, "b)")
    ####### RMS hi and lo
    ax = axs[2]
    ax.errorbar(
        cbins, datacount_list[3], yerr=(poisson_list[3]), fmt="^", c="k", label="REAL DATA HIGH"
    )
    ax.plot(cbins, simcount_list[3], c="tab:orange", label="SIMULATION HI", ls="--")
    ax.errorbar(
        cbins,
        datacount_list[4],
        yerr=(poisson_list[4]),
        fmt="s",
        c="tab:green",
        label="REAL DATA LOW",
    )
    ax.plot(cbins, simcount_list[4], c="tab:blue", label="SIMULATION LOW ")
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$\sigma_{\rm r}$")
    thestring = r"High $\chi^2_{\sigma_{\rm r}} =$ " + str(np.around(chisq[3], 1))
    ax.text(
        -0.2,
        0.42,
        thestring,
    )
    thestring = r"Low $\chi^2_{\sigma_{\rm r}} =$ " + str(np.around(chisq[4], 1))
    ax.text(
        -0.2,
        0.395,
        thestring,
    )
    ax.text(-0.2, 0.48, "c)")
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
    plt.savefig(
        OUTDIR
        + "figures/"
        + DATA_INPUT.split(".")[0].split("/")[-1]
        + "overplot_observed_DATA_SIM_OVERVIEW.pdf",
        pad_inches=0.01,
        bbox_inches="tight",
    )
    print("upload " + OUTDIR + "figures/overplot_observed_DATA_SIM_OVERVIEW.pdf")
    plt.close()
    return "Done"
    # END Criteria_Plotter


def subprocess_to_snana(OUTDIR, snanadict):
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
    for i in snanadict.keys():
        if i in filedata:
            filedata = filedata.replace(i, snanadict[i])
    os.remove(filein)
    f = open(filein, "w")
    f.write(filedata)
    f.close()
    return "Done"
    # END subprocess_to_snana


# =======================================================
################### CONNCECTIONS #######################
# =======================================================


def init_connection(index, real=True, debug=False, CMD_DATA=None, cmd_sim=None):
    # Creates an open connection instance with SALT2mu.exe
    # def OPTMASK=1 Default. Creates a FITRES file.
    # def OPTMASK=2 Creates an M0DIF file
    # def OPTMASK=4 implements randomseed option

    OPTMASK = 4
    directory = "parallel"
    if DEBUG:
        OPTMASK = 1
    elif SINGLE:
        OPTMASK = 1

    realdataout = f"{OUTDIR}{directory}/%d_SUBPROCESS_REALDATA_OUT.DAT" % index
    Path(realdataout).touch()
    simdataout = f"{OUTDIR}{directory}/%d_SUBROCESS_SIM_OUT.DAT" % index
    Path(simdataout).touch()
    mapsout = f"{OUTDIR}{directory}/%d_PYTHONCROSSTALK_OUT.DAT" % index
    Path(mapsout).touch()
    subprocess_log_data = f"{OUTDIR}{directory}/%d_SUBPROCESS_LOG_DATA.STDOUT" % index
    Path(subprocess_log_data).touch()
    subprocess_log_sim = f"{OUTDIR}{directory}/%d_SUBPROCESS_LOG_SIM.STDOUT" % index
    Path(subprocess_log_sim).touch()
    arg_outtable = f"'c(6,-0.2:0.25)*{splitparamdict[SPLITPARAM]}'"  # need to programmatically generate the second option
    GENPDF_NAMES = f"SIM_x1,{SPLITPARAM},SIM_c,SIM_RV,SIM_EBV,SIM_ZCMB,SIM_beta"  # need to programmatically generate the split
    if real:
        cmd = (
            f"{JOBNAME_SALT2mu} {DATA_INPUT} "
            f"SUBPROCESS_FILES=%s,%s,%s "
            f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} "
            f"debug_flag=930"
        )
        if CMD_DATA:
            cmd = cmd + " {CMD_DATA}"
        if OPTMASK < 4:
            cmd = cmd + f" SUBPROCESS_OPTMASK={OPTMASK}"
        realdata = callSALT2mu.SALT2mu(
            cmd,
            OUTDIR + "NOTHING.DAT",
            realdataout,
            subprocess_log_data,
            realdata=True,
            debug=DEBUG,
        )

    else:
        realdata = 0
    cmd = (
        f"{JOBNAME_SALT2mu} {SIM_INPUT} SUBPROCESS_FILES=%s,%s,%s "
        f"SUBPROCESS_VARNAMES_GENPDF={GENPDF_NAMES} "
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} "
        f"SUBPROCESS_OPTMASK={OPTMASK} "
        f"SUBPROCESS_SIMREF_FILE={SIMREF_FILE} "
        f"debug_flag=930"
    )
    if cmd_sim:
        cmd = cmd + " {cmd_sim}"
    connection = callSALT2mu.SALT2mu(cmd, mapsout, simdataout, subprocess_log_sim, debug=DEBUG)
    if not real:  # connection is an object that is equal to SUBPROCESS_SIM/DATA
        connection.getResult()  # Gets result, as it were
    return realdata, connection
    # END init_connection


def connection_prepare(
    connection,
):  # probably works. Iteration issues, needs to line up with SALT2mu and such.
    connection.iter += 1  # tick up iteration by one
    connection.write_iterbegin()  # open SOMETHING.DAT for that iteration
    return connection
    # END connection_prepare


def connection_next(connection):  # Happens at the end of each iteration.
    connection.write_iterend()
    print("wrote end")
    connection.next()
    print("submitted next iter")
    connection.getResult()
    return connection
    # END connection_next


def normhisttodata(datacount, simcount):
    # Helper function to
    # normalize the simulated histogram to the total counts of the data
    datacount = np.array(datacount)
    simcount = np.array(simcount)
    datatot = np.sum(datacount)
    simtot = np.sum(simcount)
    simcount = simcount * datatot / simtot

    ww = (datacount != 0) | (simcount != 0)

    # poisson = np.sqrt(datacount)        commented out to switch to poisson errors from sim
    # poisson[datacount == 0] = 1            commented out to switch to poisson errors from sim
    # poisson[~np.isfinite(poisson)] = 1   commented out to switch to poisson errors from sim
    poisson = np.sqrt(simcount)
    poisson[simcount == 0] = 1
    poisson[~np.isfinite(poisson)] = 1
    return datacount[ww], simcount[ww], poisson[ww], ww
    # END normhisttodata


# =======================================================
################### SCIENCE FUNCTIONS ##################
# =======================================================


# this uses the built in nautilus function to make prior by successively adding parameters to a class wrapper
# for simplicity, we are going to ignore the first three parameters of CLEANDICT (loc, scale and IsPositive)
# and just use a uniform prior specified by the 2 params in entry 3
def make_nautilus_prior(INP_PARAMS, CLEANDICT, override, walkfactor=2):
    plist = pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT)
    for element in override.keys():
        plist.remove(element)
    prior = nautilus.Prior()
    for p in plist:
        param = CLEANDICT[p]
        mu = param[0]
        sigma = param[1]
        lower, upper = param[3][0], param[3][1]
        prior.add_parameter(
            p, dist=truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        )
    return prior


# this is just bounds checking which is not needed for bounded priors used in nested sampling
# def log_prior(theta, debug=False): #goes through expanded input parameters and checks that they are all within range. If any are not, returns negative infinity.
#     thetadict = thetaconverter(theta)
#     plist = pconv(INP_PARAMS,PARAMSHAPESDICT, SPLITDICT)
#     if DEBUG: print('plist', plist)
#     tlist = False #if all parameters are good, this remains false
#     for key in thetadict.keys():
#         if DEBUG: print('key', key)
#         temp_ps = (thetawriter(theta, key)) #I hate this but it works. Creates expanded list for this parameter
#         if DEBUG: print('temp_ps', temp_ps)
#         plist_n = (thetawriter(theta, key, names=plist))
#         for t in range(len(temp_ps)): #then goes through
#             if DEBUG: print('plist name', plist_n[t])
#             lowb = CLEANDICT[plist_n[t]][3][0]
#             highb = CLEANDICT[plist_n[t]][3][1]
#             if DEBUG: print(lowb, temp_ps[t], highb)
#             if  not lowb < temp_ps[t] < highb: # and compares to valid boundaries.
#                 tlist = True
#     if tlist:
#         return -np.inf
#     else:
#         return 0
# END log_prior


def log_likelihood(theta, connection=False, returnall=False, pid=0, genpdf_only=False):
    splitEBV, stretch = True, False

    thetadict = thetaconverter(theta)
    # for elem in override.keys():
    #    print(elem, flush=True)
    #    position = (pconv(INP_PARAMS).index(elem),paramshapesdict.index(elem), splitdict.index(elem))
    #    theta.insert(position, override[elem])
    # here try and use theta.insert(position, element) to put the override value back where we expect it to be
    try:
        if connection == False:  # For MCMC running, will pick up a connection
            sys.stdout.flush()
            connection = connections[
                (current_process - 1)
            ]  # formerly connections[(current_process()._identity[0]-1) % len(connections)]
            print(f"Current PID is {os.getpid()}")
            sys.stdout.flush()
        connection = connection_prepare(connection)  # cycle iteration, open SOMETHING.DAT
        print("writing 1d pdf", flush=True)
        for inp in INP_PARAMS:  # TODO - need to generalise to 2d functions as well
            connection.write_generic_PDF(
                inp,
                SPLITDICT,
                thetawriter(theta, inp),
                PARAMSHAPESDICT[inp],
                shapedict,
                simdict,
                array_conv(inp, SPLITDICT, SPLITARR),
            )

        if genpdf_only:
            print("GENPDF File created. Quitting now. This is expected.")
            return

        print("next", flush=True)
        # AAAAAAAA
        connection = connection_next(connection)  # NOW RUN SALT2mu with these new distributions
        print("got result", flush=True)
        # AAAAAAAAA
        try:
            if connection.maxprob > 1.001:
                print(
                    connection.maxprob,
                    "MAXPROB parameter greater than 1! Coming up against the bounding function! Returning -np.inf to account, caught right after connection",
                    flush=True,
                )
                return -np.inf
        except AttributeError:
            print(
                "Can't find MAXPROB value. This may be a bad parameter or a fluke of some sort. Will return negative infinity"
            )
            return -np.inf
        try:
            if np.isnan(connection.beta):
                print("WARNING! oops negative infinity!")
                newcon = (
                    current_process - 1
                )  # % see above at original connection generator, this has been changed
                tc = init_connection(newcon, real=False)[1]
                connections[newcon] = tc
                return -np.inf
        except AttributeError:
            print("WARNING! We tripped an AttributeError here.")
            if DEBUG:
                print("inp", inp)
                return -np.inf
            else:
                newcon = (
                    current_process - 1
                )  # % see above at original connection generator, this has been changed
                tc = init_connection(newcon, real=False)[1]
                connections[newcon] = tc
                return -np.inf
        # ANALYSIS returns c, highres, lowres, rms
        print("Right before calculation", flush=True)
        try:
            bindf = connection.bindf  # THIS IS THE PANDAS DATAFRAME OF THE OUTPUT FROM SALT2mu
            bindf = bindf.dropna()
            sim_vals = dffixer(bindf, "ANALYSIS", False)
            realbindf = realdata.bindf  # same for the real data (was a global variable)
            realbindf = realbindf.dropna()
            real_vals = dffixer(realbindf, "ANALYSIS", True)
            resparr = []
            for lin in range(len(real_vals)):
                resparr.append([real_vals[lin], sim_vals[lin]])
        except Exception as e:
            print(e)
            print("WARNING! something went wrong in reading in stuff for the LL calc")
            return -np.inf
    except BrokenPipeError:
        if DEBUG:
            print("WARNING! we landed in a Broken Pipe error")
            quit()
        else:
            print("WARNING! Slurm Broken Pipe Error!")  # REGENERATE THE CONNECTION
            print("before regenerating")
            newcon = (
                current_process - 1
            )  # % see above at original connection generator, this has been changed
            tc = init_connection(newcon, real=False)[1]
            connections[newcon] = tc
            return log_likelihood(theta, connection=tc)
    sys.stdout.flush()
    print("Right before calling LL Creator", flush=True)
    if returnall:
        out_result = LL_Creator(resparr, connection.beta, connection.sigint)
        print(
            "for ",
            pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT),
            theta,
            " we found an LL of",
            out_result,
        )
        sys.stdout.flush()
        return LL_Creator(resparr, connection.beta, connection.sigint, returnall)
    else:
        out_result = LL_Creator(resparr, connection.beta, connection.sigint)
        print(
            "for ",
            pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT),
            " parameters = ",
            theta,
            "we found an LL of",
            out_result,
        )
        sys.stdout.flush()
        return out_result
    # END log_likelihood


# this removes the bounds checking on the prior (unnecessary as we are using bounded priors)
def log_probability(theta):
    # lp = log_prior(theta)
    # if not np.isfinite(lp):
    #     print('WARNING! We returned -inf from small parameters!')
    #     sys.stdout.flush()
    #     return -np.inf
    sys.stdout.flush()
    return log_likelihood(theta)
    # END log_probability


def init_dust2dust():
    global realdata
    if DEBUG or SINGLE or DOPLOT:
        realdata, _ = init_connection(299, debug=DEBUG)
    else:
        realdata, _ = init_connection(0, debug=DEBUG)
    #########################################################
    realbeta = realdata.beta
    realbetaerr = realdata.betaerr
    realsigint = realdata.sigint
    realsiginterr = 0.0036
    #########################################################
    return realbeta, realbetaerr, realsigint, realsiginterr
    # END init_dust2dust


def init_connections(npool):  # changed this to npool
    connections = []
    if DEBUG:
        print("we are in debug mode now")
        npool = 1
    else:
        pass
    nconn = npool
    for i in range(int(nconn)):
        print("generated", i, "connection for the pool.")
        sys.stdout.flush()
        tc = init_connection(i, real=False, debug=DEBUG)[1]  # set this back to DEBUG=DEBUG
        connections.append(tc)
    print("Done initialising the pool connections.")
    return connections
    # END init_connections


# def MCMC(nwalkers,ndim, mpi):
#     with Pool(nwalkers) as pool: #Should I be specifying the number of walkers when instancing pool?
#         #Instantiate the sampler once (in parallel)
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
#         for qb in range(50):
#             print("Starting loop iteration", qb)
#             print('begun', cpu_count(), "CPUs with", nwalkers, ndim, "walkers and dimensions")
#             sys.stdout.flush()
#             #Run the sampler
#             if qb == 0:
#                 state2 = sampler.run_mcmc(pos, 100, progress=True)
#             else:
#                 state2 = sampler.run_mcmc(None, 300, progress=True)
#             sys.stdout.flush()
#             #Save the output for later
#             samples = sampler.get_chain()
#             np.savez(OUTDIR+'chains/'+DATA_INPUT.split('.')[0].split('/')[-1]+'-samples.npz',samples)
#             #pltting_func(samples, INP_PARAMS, ndim)
#     return "Hi!"
# END MCMC


def do_sampling(
    prior,
    npool,
    chainfile,
    n_like_max=100000,
    ndim=None,
    nlive=None,
    n_eff=10000,
    fileout=None,
    resume=True,
    seed=None,
    discard_exploration=False,
):
    if ndim == None:
        ndim = prior.dimensionality()
    if nlive == None:
        nlive = ndim * 10
        print("You didn't specify an nlive so I'll use ", nlive)
    if resume == True:
        print("Resuming sampling from file ", fileout)
    print(
        "GO! Running sampler with ndim=",
        ndim,
        "n_eff=",
        n_eff,
        "n_like_max=",
        n_like_max,
        "nlive=",
        nlive,
        "checkpoint=",
        fileout,
        "will save chains to ",
        chainfile,
    )
    #    with Pool(npool) as pool:
    sampler = nautilus.Sampler(
        prior,
        log_probability,
        n_live=nlive,
        n_dim=ndim,
        pass_dict=False,
        filepath=fileout,
        resume=resume,
        seed=seed,
    )
    sampler.run(
        verbose=True, n_eff=n_eff, n_like_max=n_like_max, discard_exploration=discard_exploration
    )
    sys.stdout.flush()
    print("Finished!")
    print("Saving chains to ", chainfile)
    # Save the output : unfortunately there doesn't seem to be a straightforward function in nautilus to save as you progress.
    points, log_w, log_l = sampler.posterior()

    np.savez(chainfile, points, log_w, log_l)
    return "I've finished!"


# =================================================================================================
###############################
# =================================================================================================

# if __name__ == "__main__":
#    try:
#        setup_logging()
#        logging.info("# ========== BEGIN DUST2DUST ===============")
args = get_args()
config = load_config(args.CONFIG)
(
    DATA_INPUT,
    INP_PARAMS,
    OUTDIR,
    SINGLE,
    DEBUG,
    NOWEIGHT,
    DOPLOT,
    CHAINS,
    CMD_DATA,
    CMD_SIM,
    PARAMS,
    SIM_INPUT,
    PARAMSHAPESDICT,
    SPLITDICT,
    CLEANDICT,
    SPLITARR,
    SIMREF_FILE,
    GENPDF_ONLY,
    SPLITPARAM,
) = prep_config(args, config)
# Run Main Code here
# pos, nwalkers, ndim = input_cleaner(INP_PARAMS, CLEANDICT,override, walkfactor=3) # replaced by make_nautilus_prior
prior = make_nautilus_prior(INP_PARAMS, CLEANDICT, override)
ndim = prior.dimensionality()
realbeta, realbetaerr, realsigint, realsiginterr = init_dust2dust()
# everything that is not the MCMC happens between here...
if SINGLE:
    if len(PARAMS) != ndim:
        print(
            f"Your input parameters ({len(PARAMS)}) are configured incorrectly, and do not match the expected number of parameters ({ndim}). Quitting to avoid confusion."
        )
        quit()
    print("inp", PARAMS)
    Criteria_Plotter(PARAMS)
    quit()
elif GENPDF_ONLY:
    print("Generating GENPDF now...")
    if len(PARAMS) != ndim:
        print(
            f"Your input parameters ({len(PARAMS)}) are configured incorrectly, and do not match the expected number of parameters ({ndim}). Quitting to avoid confusion."
        )
        quit()
    print("inp", PARAMS)
    Criteria_Plotter(PARAMS, genpdf_only=True)
    subprocess_to_snana(OUTDIR, snanadict)
    quit()
elif DOPLOT:
    if not CHAINS:
        print("Please specify a path to the chains file you wish to plot. Quitting gracefully.")
        quit()
    print("DOPLOT option enabled. Plotting now", flush=True)
    old_chains = np.load(CHAINS)
    past_results = old_chains.f.arr_0
    Criteria_Plotter(PARAMS)
    pltting_func(past_results, PARAMS, ndim)
    # ... and here.
    # Initialise the MCMC-exclusive parts of the code.
connections = init_connections(npool)
# MCMC(nwalkers,ndim, False)
do_sampling(
    prior,
    npool,
    CHAINS,
    n_like_max=n_like_max,
    n_eff=n_eff,
    fileout=checkpoint,
    resume=resume,
    seed=seed,
    discard_exploration=discard_exploration,
)

# except Exception as e:
#    logging.exception(e)
#    raise e
