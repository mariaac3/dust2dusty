"""
callSALT2mu: Interface to SALT2mu.exe for supernova cosmology analysis

This module provides a Python interface to the SALT2mu.exe C executable,
enabling subprocess-based communication for iterative likelihood evaluations.

Key features:
    - SALT2mu class: Manages persistent subprocess connection to SALT2mu.exe
    - PDF generation: Writes probability distribution functions for reweighting
    - Result parsing: Extracts fit parameters (alpha, beta, sigint) and binned data
    - Support for 1D, 2D, and 3D PDFs with arbitrary splits

The SALT2mu class maintains a bidirectional communication with the SALT2mu.exe
subprocess through files:
    - mapsout: Python writes PDF functions here (input to SALT2mu)
    - SALT2muout: SALT2mu writes fit results here (output from SALT2mu)
    - Process stdin/stdout: Used for iteration control

Typical workflow:
    1. Initialize SALT2mu object (launches subprocess)
    2. For each MCMC iteration:
       a. Prepare iteration (increment counter, open file)
       b. Write PDF functions for each parameter
       c. Close iteration (flush file)
       d. Send iteration number to subprocess
       e. Parse results from output file
"""

import logging
import os
import subprocess
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd


def setup_custom_logger(name, screen=False, debug=False):
    """
    Create custom logger with file and optional screen output.

    Args:
        name: Logger name (also used for log filename)
        screen: If True, also log to stdout (default: False)

    Returns:
        logging.Logger: Configured logger instance
                        Logs to logs/log_{name}.log with DEBUG level
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.FileHandler(f"logs/log_{name}.log", mode="a+")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    if screen:
        logger.addHandler(screen_handler)
    logger.setLevel(logging.DEBUG)
    if not debug:
        logger.addHandler(logging.NullHandler())

    return logger
    # END setup_custom_logger


class SALT2mu:
    """
    Interface class for SALT2mu.exe subprocess communication.

    Manages persistent connection to SALT2mu.exe, handling PDF writing,
    subprocess control, and result parsing.

    Attributes:
        logger: Custom logger for this instance
        iter: Current iteration number (starts at -1)
        debug: Debug mode flag
        ready, ready2, done, initready: Subprocess status strings
        crosstalkfile: File handle for writing PDFs (mapsout)
        SALT2muoutputs: File handle for reading results (SALT2muout)
        command: Command string to launch SALT2mu.exe
        process: Subprocess object (if realdata=False)
        stdout_iterator: Iterator for subprocess stdout
        alpha, alphaerr: Alpha parameter and error from SALT2mu fit
        beta, betaerr: Beta parameter and error from SALT2mu fit
        maxprob: Maximum probability ratio (for bounding function check)
        sigint: Intrinsic scatter from SALT2mu fit
        bindf: pandas DataFrame with binned SALT2mu output
    """

    def __init__(self, command, mapsout, SALT2muout, log, realdata=False, debug=False):
        """
        Initialize SALT2mu connection.

        Args:
            command: Command string for SALT2mu.exe with %s placeholders for files
                     Format: "SALT2mu.exe input.file SUBPROCESS_FILES=%s,%s,%s ..."
            mapsout: Path for PDF crosstalk file (Python writes, SALT2mu reads)
            SALT2muout: Path for results file (SALT2mu writes, Python reads)
            log: Path for subprocess log file
            realdata: If True, run synchronously and return immediately (default: False)
            debug: If True, enable debug logging and YAML output (default: False)

        Side effects:
            - Creates empty files at mapsout and SALT2muout paths
            - If realdata=True: Runs SALT2mu via os.system and calls getData()
            - If realdata=False: Launches SALT2mu.exe subprocess
        """
        print("COMMAND:", command % (mapsout, SALT2muout, log), flush=True)

        self.logger = setup_custom_logger(
            "walker_" + os.path.basename(mapsout).split("_")[0], debug=debug
        )
        self.iter = -1
        self.debug = debug  # Boolean. Default False.
        self.ready_enditer = "Enter expected ITERATION number"
        self.ready2 = "ITERATION=%d"
        self.done = "Graceful Program Exit. Bye."
        self.initready = "Finished SUBPROCESS_INIT"
        self.crosstalkfile = open(mapsout, "w")
        self.SALT2muoutputs = open(SALT2muout, "r")  # An output file

        self.command = command % (mapsout, SALT2muout, log)

        self.logger.info("Init SALT2mu instance. ")
        self.logger.info("## ================================== ##")
        self.logger.info(f"Command: {self.command}")
        self.logger.info(f"mapsout: {mapsout}")
        self.logger.info(f"Debug mode={self.debug}")

        if self.debug:
            self.command = self.command + " write_yaml=1"

        self.logger.info("Command being run: " + self.command)
        self.data = False

        self.process = subprocess.Popen(
            command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.wait_until_text_in_output(self.ready_enditer)

        if realdata:  # this is awful )
            self.logger.info("Running realdata=True")
            self.getData()  # calls getData
            self.quit()
        # END __init__

    def quit(self):  # sets iteration input to -1, which causes a quit somewhere
        self.process.stdin.write("-1\n")
        for stdout_line in iter(self.process.stdout.readline, ""):
            print(stdout_line)
        # END quit
        #

    def wait_until_text_in_output(self, expected_text, timeout=120):
        start = time.time()

        # Wait for specific output using iter()
        for line in iter(self.process.stdout.readline, ""):
            if expected_text in line:
                break

            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for '{expected_text}'")
        return

    def next_iter(
        self,
        theta_dic,
        config,
    ):  # ticks up iteration by one
        self.iter += 1
        self.write_iterbegin()

        for par_key in config.inp_params:
            bounds = (theta_dic[par_key][0], theta_dic[par_key][-1] + 1)

            arr = []
            if par_key not in ["beta", "alpha"]:
                arr.append(config.DEFAULT_PARAMETER_RANGES[par_key])
                if par_key in config.splitdict.keys():
                    for s in config.splitdict[par_key].keys():
                        arr.append(eval((config.splitarr[s])))

            self.write_generic_PDF(
                par_key,
                config.splitdict,
                bounds,
                config.paramshapesdict[par_key],
                config.DISTRIBUTION_PARAMETERS,
                config.PARAM_TO_SALT2MU,
                arr,
            )

        self.logger.info("Launch next step")
        self.write_iterend()

        # Launch SALT2mu on new dist and wait for done
        self.process.stdin.write("%d\n" % self.iter)
        self.process.stdin.flush()
        self.wait_until_text_in_output(self.ready_enditer)

        self.data = self.getData()
        # END next

    def getData(self):
        """
        Parse SALT2mu output file to extract fit results.

        Reads SALT2muout file and extracts:
        - alpha, alphaerr: SALT2 standardization parameter
        - beta, betaerr: Color-luminosity parameter
        - maxprob: Maximum probability ratio (for boundary checking)
        - sigint: Intrinsic scatter
        - bindf: pandas DataFrame with binned statistics (color, mass, MURES, RMS)

        Returns:
            bool: True upon successful parsing

        Side effects:
            - Sets self.alpha, self.alphaerr, self.beta, self.betaerr,
              self.maxprob, self.sigint, self.bindf, self.headerinfo
        """
        self.SALT2muoutputs.seek(0)  # sets pointer to top of file
        text = self.SALT2muoutputs.read()  # reads in the text
        self.alpha = float(text.split("alpha0")[1].split()[1])
        self.alphaerr = float(text.split("alpha0")[1].split()[3])
        self.beta = float(text.split("beta0")[1].split()[1])
        self.betaerr = float(text.split("beta0")[1].split()[3])
        self.maxprob = float(text.split("MAXPROB_RATIO")[1].split()[1])
        self.headerinfo = self.NAndR(StringIO(text))
        self.sigint = float(text.split("sigint")[1].split()[1])
        self.bindf = pd.read_csv(
            StringIO(text),
            header=None,
            skiprows=self.headerinfo[1],
            names=self.headerinfo[0],
            delim_whitespace=True,
            comment="#",
        )
        self.siginterr = 0.0036  # DEFAULT VALUE
        return True
        # END getData

    def get_1d_asym_gauss(self, mean, lhs, rhs, arr):
        """
        Generate asymmetric Gaussian probability distribution.

        Creates Gaussian with different widths left and right of mean.

        Args:
            mean: Central value
            lhs: Standard deviation for values < mean
            rhs: Standard deviation for values > mean
            arr: Array of x values to evaluate PDF at

        Returns:
            tuple: (arr, probs) where probs are normalized to max=1
        """
        probs = np.exp(-0.5 * ((arr - mean) / lhs) ** 2)
        probs[arr > mean] = np.exp(-0.5 * ((arr[arr > mean] - mean) / rhs) ** 2)
        probs = probs / np.max(probs)
        return arr, probs  # x and y
        # END get_1d_asym_gauss

    def get_1d_exponential(self, tau, arr):
        """
        Generate exponential probability distribution.

        Args:
            tau: Exponential decay constant (scale parameter)
            arr: Array of x values to evaluate PDF at (should be >= 0)

        Returns:
            tuple: (arr, probs) where probs are normalized to max=1
        """
        probs = (tau**-1) * np.exp(-arr / tau)
        probs = probs / np.max(probs)
        return arr, probs
        # END get_1d_exponential

    def get_1d_lognormal(self, mu, std, arr):
        """
        Generate log-normal probability distribution.

        NOTE: Current implementation appears incorrect - should use
        log-normal formula, not exp(mu + std*arr).

        Args:
            mu: Location parameter
            std: Scale parameter
            arr: Array of x values to evaluate PDF at

        Returns:
            tuple: (arr, probs) where probs are normalized to max=1
        """
        probs = np.exp(mu + std * arr)
        probs = probs / np.max(probs)
        return arr, probs
        # END get_1d_lognormal

    def writeheader(self, names):  # writes the header - DEPRECATED
        self.crosstalkfile.write("VARNAMES:")
        for name in names:
            self.crosstalkfile.write(" " + name)
        self.crosstalkfile.write(" PROB\n")
        return
        # END writeheader

    def writegenericheader(self, inp, varnames):
        """
        Write VARNAMES header line for PDF block.

        Args:
            inp: Main variable name (e.g., 'SIM_c', 'SIM_RV')
            varnames: List of additional variable names for splits
                      (e.g., ['HOST_LOGMASS'] for 2D PDF)

        Side effects:
            - Writes "VARNAMES: inp varname1 varname2 ... PROB\n" to crosstalk file
        """
        self.crosstalkfile.write(f"VARNAMES: {inp}")
        for name in varnames:
            self.crosstalkfile.write(" " + name)
        self.crosstalkfile.write(" PROB \n")
        return
        # END writegenericheader

    def write3Dprobs(self, arr, z, mass, probs):
        """
        Write 3D probability distribution to crosstalk file.

        Format: "PDF: value z mass prob" for each point

        Args:
            arr: Array of primary variable values
            z: Redshift value (scalar)
            mass: Mass value (scalar)
            probs: Array of probability values (same length as arr)

        Side effects:
            - Writes PDF lines to crosstalk file
        """
        bigstr = ""
        for a, p in zip(arr, probs):
            bigstr += "PDF: %.3f %.2f %.2f %.3f\n" % (a, z, mass, p)
        self.crosstalkfile.write(bigstr)
        # END write3Dprobs

    def write2Dprobs(self, arr, mass, probs):
        """
        Write 2D probability distribution to crosstalk file.

        Format: "PDF: value mass prob" for each point

        Args:
            arr: Array of primary variable values
            mass: Mass value (scalar)
            probs: Array of probability values (same length as arr)

        Side effects:
            - Writes PDF lines to crosstalk file
        """
        bigstr = ""
        for a, p in zip(arr, probs):
            bigstr += "PDF: %.3f %.2f %.3f\n" % (a, mass, p)
        self.crosstalkfile.write(bigstr)
        # END write2Dprobs

    def write1Dprobs(self, arr, probs):
        """
        Write 1D probability distribution to crosstalk file.

        Format: "PDF: value prob" for each point

        Args:
            arr: Array of variable values
            probs: Array of probability values (same length as arr)

        Side effects:
            - Writes PDF lines to crosstalk file
            - Adds blank line at end
        """
        bigstr = ""
        for a, p in zip(arr, probs):
            bigstr += "PDF: %.3f %.3f\n" % (a, p)
        bigstr += "\n"
        self.crosstalkfile.write(bigstr)
        # END write1Dprobs

    def write_iterbegin(self):
        """
        Start new iteration in crosstalk file.

        Clears file and writes iteration begin marker.

        Side effects:
            - Truncates crosstalk file to zero length
            - Writes "ITERATION_BEGIN: N" where N = self.iter
        """
        self.crosstalkfile.truncate(0)
        self.crosstalkfile.seek(0)
        self.crosstalkfile.write("ITERATION_BEGIN: %d\n" % self.iter)
        # END write_iterbegin

    def write_iterend(self):
        """
        Mark end of iteration in crosstalk file.

        Side effects:
            - Writes "ITERATION_END: N" where N = self.iter
            - Flushes crosstalk file to ensure SALT2mu can read it
        """
        self.crosstalkfile.write("ITERATION_END: %d\n" % self.iter)
        self.crosstalkfile.flush()
        # END write_iterend

    def write_SALT2(self, name, PARAMS):
        """
        Write SALT2 standardization parameters (alpha/beta) in SNANA format.

        Alpha and beta are handled differently from other parameters - they use
        SNANA GENPEAK/GENSIGMA/GENRANGE format instead of PDF format.

        Args:
            name: Parameter name ('alpha' or 'beta')
            PARAMS: List [mean, std] for Gaussian distribution

        Side effects:
            - Writes SNANA-format parameter specification to crosstalk file
            - GENRANGE: .1-.2 for alpha, .4-3 for beta
        """
        for tm in range(3):
            self.crosstalkfile.write("\n")
        mean = PARAMS[0]
        std = PARAMS[1]
        self.crosstalkfile.write(f"GENPEAK_SIM_{name}: {mean} \n")
        self.crosstalkfile.write(f"GENSIGMA_SIM_{name}: {std} {std} \n")
        if name == "alpha":
            self.crosstalkfile.write(f"GENRANGE_SIM_{name}: .1 .2 \n")
        else:
            self.crosstalkfile.write(f"GENRANGE_SIM_{name}: .4 3 \n")
        for tm in range(3):
            self.crosstalkfile.write("\n")
        # END write_SALT2

    def write_1D_PDF(self, varname, PARAMS, arr):  # DEPRECATED
        self.writeheader([varname])
        try:
            mean, lhs, rhs = PARAMS
        except ValueError:
            mean = PARAMS[0]
            lhs = PARAMS[1]
            rhs = lhs
        arr, probs = self.get_1d_asym_gauss(mean, lhs, rhs, arr)
        self.write1Dprobs(arr, probs)
        # END write_1D_PDF

    def writeRVLOGMASS_PDF(self, mean, lhs, rhs):  # DEPRECATED
        self.writeheader(["RV", "LOGMASS"])
        for mass in massarr:
            if mass < 10:
                arr, probs = self.get_1d_asym_gauss(mean, lhs, rhs)
                self.write2Dprobs(arr, mass, probs)
            else:
                arr, probs = self.get_1d_asym_gauss(mean, lhs, rhs)
            probs[arr < 0.5] = 0
            self.write2Dprobs(arr, mass, probs)
        # END writeRVLOGMASS_PDF

    def write_2D_Mass_PDF(self, varname, PARAMS, arr):  # DEPRECATED
        self.writeheader([varname, "HOST_LOGMASS"])
        Lmean, Llhs, Lrhs = PARAMS[0:3]
        Hmean, Hlhs, Hrhs = PARAMS[3:]
        for mass in massarr:
            if mass < 10:
                arr, probs = self.get_1d_asym_gauss(Lmean, Llhs, Lrhs, arr)
                probs[arr < 0.4] = 0
                self.write2Dprobs(arr, mass, probs)
            else:
                arr, probs = self.get_1d_asym_gauss(Hmean, Hlhs, Hrhs, arr)
                probs[arr < 0.4] = 0
                self.write2Dprobs(arr, mass, probs)
        self.crosstalkfile.write("\n")
        # END write_2D_Mass_PDF

    def write_2D_Mass_PDF_SYMMETRIC(self, varname, PARAMS, arr):  # DEPRECATED
        self.writeheader([varname, "HOST_LOGMASS"])
        Lmean, Llhs = PARAMS[0:2]
        Lrhs = Llhs
        Hmean, Hlhs = PARAMS[2:]
        Hrhs = Hlhs
        for mass in massarr:
            if mass < 10:
                arr, probs = self.get_1d_asym_gauss(Lmean, Llhs, Lrhs, arr)
                probs[arr < 0.4] = 0
                self.write2Dprobs(arr, mass, probs)
            else:
                arr, probs = self.get_1d_asym_gauss(Hmean, Hlhs, Hrhs, arr)
                probs[arr < 0.4] = 0
                self.write2Dprobs(arr, mass, probs)
        self.crosstalkfile.write("\n")
        # END write_2D_Mass_PDF_SYMMETRIC

    def write_2D_MassEBV_PDF(self, varname, PARAMS, arr):  # DEPRECATED
        self.writeheader([varname, "HOST_LOGMASS"])
        LTau = PARAMS[0]
        HTau = PARAMS[1]
        for mass in massarr:
            if mass < 10:
                arr, probs = self.get_1d_exponential(LTau, arr)
                self.write2Dprobs(arr, mass, probs)
            else:
                arr, probs = self.get_1d_exponential(HTau, arr)
                self.write2Dprobs(arr, mass, probs)
        self.crosstalkfile.write("\n")
        # END write_2D_MassEBV_PDF

    def write_2D_LOGNORMAL_PDF(self, varname, PARAMS, arr):  # DEPRECATED
        self.writeheader([varname, "HOST_LOGMASS"])
        Lmu, Lstd = PARAMS[0, 1]
        Hmu, Hstd = PARAMS[2, 3]
        for mass in massarr:
            if mass < 10:
                arr, probs = self.get_1d_lognormal(Lmu, Lstd, arr)
                self.write2Dprobs(arr, mass, probs)
            else:
                arr, probs = self.get_1d_lognormal(Hmu, Hstd, arr)
                self.write2Dprobs(arr, mass, probs)
        self.crosstalkfile.write("\n")
        # END write_2D_LOGNORMAL_PDF

    def write_3D_MassEBV_PDF(
        self, varname, PARAMS, arr
    ):  # for when EBV needs a z split - DEPRECATED
        self.writeheader([varname, "SIM_ZCMB", "HOST_LOGMASS"])  # needs work
        LZ_LTau = PARAMS[0]
        LZ_HTau = PARAMS[1]
        HZ_LTau = PARAMS[2]
        HZ_HTau = PARAMS[3]
        for z in zarr:
            if np.around(z, 3) < 0.1:
                for mass in massarr:
                    if mass < 10:
                        arr, probs = self.get_1d_exponential(LZ_LTau, arr)
                        self.write3Dprobs(arr, z, mass, probs)
                    else:
                        arr, probs = self.get_1d_exponential(LZ_HTau, arr)
                        self.write3Dprobs(arr, z, mass, probs)
            else:
                for mass in massarr:
                    if mass < 10:
                        arr, probs = self.get_1d_exponential(HZ_LTau, arr)
                        self.write3Dprobs(arr, z, mass, probs)
                    else:
                        arr, probs = self.get_1d_exponential(HZ_HTau, arr)
                        self.write3Dprobs(arr, z, mass, probs)
        self.crosstalkfile.write("\n")
        # END write_3D_MassEBV_PDF

    def write_3D_LOGNORMAL_PDF(
        self, varname, PARAMS, arr
    ):  # for when EBV needs a z split - DEPRECATED
        self.writeheader([varname, "SIM_ZCMB", "HOST_LOGMASS"])  # needs work
        LZ_Lmu, LZ_Lstd, LZ_Hmu, LZ_Hstd, HZ_Lmu, HZ_Lstd, HZ_Hmu, HZ_Hstd = PARAMS
        for z in zarr:
            if np.around(z, 3) < 0.1:
                for mass in massarr:
                    if mass < 10:
                        arr, probs = self.get_1d_lognormal(LZ_Lmu, LZ_Lstd, arr)
                        self.write3Dprobs(arr, z, mass, probs)
                    else:
                        arr, probs = self.get_1d_lognormal(LZ_Hmu, LZ_Hstd, arr)
                        self.write3Dprobs(arr, z, mass, probs)
            else:
                for mass in massarr:
                    if mass < 10:
                        arr, probs = self.get_1d_lognormal(HZ_Lmu, HZ_Lstd, arr)
                        self.write3Dprobs(arr, z, mass, probs)
                    else:
                        arr, probs = self.get_1d_lognormal(HZ_Hmu, HZ_Hstd, arr)
                        self.write3Dprobs(arr, z, mass, probs)
        self.crosstalkfile.write("\n")
        # END write_3D_LOGNORMAL_PDF

    def write_2D_PDF(self, varname, LOWPARAMS, HIGHPARAMS, arr):  # DEPRECATED
        self.writeheader([varname[0], varname[1]])
        Lmean, Llhs, Lrhs = LOWPARAMS
        Hmean, Hlhs, Hrhs = HIGHPARAMS
        for mass in arr:
            if mass < 10:
                arr, probs = self.get_1d_asym_gauss(Lmean, Llhs, Lrhs)
            else:
                arr, probs = self.get_1d_asym_gauss(Hmean, Hlhs, Hrhs)
            probs[arr < 0.5] = 0
            self.write2Dprobs(arr, mass, probs)
        # END write_2D_PDF

    def NAndR(self, fp):
        """
        Parse SALT2mu output to find variable names and data start row.

        Searches for VARNAMES line to get column headers and identifies
        where actual data rows begin.

        Args:
            fp: File-like object (StringIO) with SALT2mu output text

        Returns:
            tuple: (Names, Startrow)
                   Names: List of column names
                   Startrow: Integer row number where data begins
        """
        for i, line in enumerate(fp):
            if line.startswith("VARNAMES:"):
                line = line.replace(",", " ")
                line = line.replace("\n", "")
                Names = line.split()
            elif line.startswith("SN") or line.startswith("ROW:") or line.startswith("GAL"):
                Startrow = i
                break
        return Names, Startrow
        # END NAndR

    def write_generic_PDF(self, inp, SPLIT, PARAMS, SHAPE, SHAPEDICT, SIMDICT, arr):
        """
        Write probability distribution function for any parameter with arbitrary splits.

        Main PDF writing function that handles:
        - 1D PDFs (no splits)
        - 2D PDFs (one split variable, e.g., mass)
        - 3D PDFs (two split variables, e.g., redshift and mass)

        Args:
            inp: Parameter name (e.g., 'c', 'RV', 'EBV')
            SPLIT: Dictionary defining splits for this parameter
                   {param: {split_var: split_value}}
            PARAMS: Array of distribution parameters from MCMC
            SHAPE: Distribution shape name ('Gaussian', 'Exponential', etc.)
            SHAPEDICT: Dictionary mapping shapes to parameter names
            SIMDICT: Dictionary mapping parameter names to SALT2mu format
            arr: List of arrays [param_values, split1_values, split2_values, ...]

        Side effects:
            - Writes header and PDF data to crosstalk file
            - For RV: sets prob=0 for values < 0.4

        Returns:
            str: 'Done' (only for alpha/beta early return)
        """
        if ("alpha" in inp) or (
            "beta" in inp
        ):  # quick check to see if it's one of the weird parameters.
            self.write_SALT2(inp, PARAMS)
            return "Done"  # Ends the function early because beta/alpha don't work as below.
        splits = []  # Creates empty array for consistency
        if inp in SPLIT.keys():
            splits = list(SPLIT[inp].keys())  # if splits exist for this inp, lists
        splits = [SIMDICT[i] for i in splits]  # convert to SNANA readable format
        splitloc = [SPLIT[inp][i] for i in splits]
        self.writegenericheader(SIMDICT[inp], splits)  # Done writing the header.
        if len(splits) == 0:
            arrs, probs = self.shape_assigner(PARAMS, SHAPE, arr[0])
            if inp == "RV":
                probs[arrs < 0.4] = 0
            self.write1Dprobs(arrs, probs)
        elif len(splits) == 1:
            SPLITPARAMS = self.shape_interpret(PARAMS, inp, SHAPE, SHAPEDICT, SPLIT)
            for sp1 in arr[1]:
                if sp1 < splitloc[0]:
                    arrs, probs = self.shape_assigner(SPLITPARAMS[0], SHAPE, arr[0])
                    if inp == "RV":
                        probs[arrs < 0.4] = 0
                    self.write2Dprobs(arrs, sp1, probs)
                else:
                    arrs, probs = self.shape_assigner(SPLITPARAMS[1], SHAPE, arr[0])
                    if inp == "RV":
                        probs[arrs < 0.4] = 0
                    self.write2Dprobs(arrs, sp1, probs)
        elif len(splits) == 2:
            SPLITPARAMS = self.shape_interpret(PARAMS, inp, SHAPE, SHAPEDICT, SPLIT)
            for sp1 in arr[1]:  # starts first split "low" version
                if np.around(sp1, 3) < splitloc[0]:  # if first split condition is met
                    for sp2 in arr[2]:  # Second split
                        if np.around(sp2, 1) < splitloc[1]:  # Start second split "low" version
                            arrs, probs = self.shape_assigner(SPLITPARAMS[0], SHAPE, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
                        else:
                            arrs, probs = self.shape_assigner(SPLITPARAMS[1], SHAPE, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
                else:  # starts first split "high" version
                    for sp2 in arr[2]:  # Start second split
                        if np.around(sp2, 1) < splitloc[1]:  # Starts second split "low" version
                            arrs, probs = self.shape_assigner(SPLITPARAMS[2], SHAPE, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
                        else:
                            arrs, probs = self.shape_assigner(SPLITPARAMS[3], SHAPE, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
        self.crosstalkfile.write("\n")
        # END write_generic_PDF

    def shape_assigner(self, PARAMS, SHAPE, arr):
        """
        Generate probability distribution based on shape and parameters.

        Dispatcher function that calls appropriate distribution generator
        based on SHAPE specification.

        Args:
            PARAMS: Distribution parameters (length depends on SHAPE)
                    Gaussian: [mu, std]
                    Exponential: [tau]
                    LogNormal: [ln_mu, ln_std]
                    Skewed Gaussian: [mu, std_left, std_right]
            SHAPE: Distribution shape name
            arr: Array of x values to evaluate PDF at

        Returns:
            tuple: (arr, probs) from appropriate distribution function
        """
        if SHAPE == "Gaussian":
            return self.get_1d_asym_gauss(PARAMS[0], PARAMS[1], PARAMS[1], arr)
        elif SHAPE == "Exponential":
            return self.get_1d_exponential(PARAMS[0], arr)
        elif SHAPE == "LogNormal":
            return self.get_1d_lognormal(PARAMS[0], PARAMS[1], arr)
        elif SHAPE == "Skewed Gaussian":
            return self.get_1d_asym_gauss(PARAMS[0], PARAMS[1], PARAMS[2], arr)
        # END shape_assigner

    def shape_interpret(self, PARAMS, inp, SHAPE, SHAPEDICT, SPLIT):
        """
        Split PARAMS array into sub-arrays for each split combination.

        For parameters with splits, PARAMS contains parameters for all
        split combinations concatenated. This splits them back out.

        Example:
            RV with mass split, Gaussian shape:
            PARAMS = [mu_low, std_low, mu_high, std_high]
            Returns: [[mu_low, std_low], [mu_high, std_high]]

        Args:
            PARAMS: Concatenated array of all distribution parameters
            inp: Parameter name
            SHAPE: Distribution shape name
            SHAPEDICT: Dictionary mapping shapes to parameter names
            SPLIT: Dictionary defining splits for this parameter

        Returns:
            list: List of parameter arrays, one per split combination
                  Length = 2^(number of split variables)
        """
        temp = []
        for i in range(len(SPLIT[inp].keys()) * 2):
            temp.append(PARAMS[0 + i * len(SHAPEDICT[SHAPE]) : (i + 1) * len(SHAPEDICT[SHAPE])])
        return temp
        # END shape_interpret
