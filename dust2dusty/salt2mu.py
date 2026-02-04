"""
SALT2mu interface for supernova cosmology analysis.

This module provides a Python interface to the SALT2mu.exe C executable,
enabling subprocess-based communication for iterative likelihood evaluations.

Key Features:
    - SALT2mu class: Manages persistent subprocess connection to SALT2mu.exe
    - PDF generation: Writes probability distribution functions for reweighting
    - Result parsing: Extracts fit parameters (alpha, beta, sigint) and binned data
    - Support for 1D, 2D, and 3D PDFs with arbitrary splits

The SALT2mu class maintains bidirectional communication with the SALT2mu.exe
subprocess through files:
    - mapsout: Python writes PDF functions here (input to SALT2mu)
    - SALT2muout: SALT2mu writes fit results here (output from SALT2mu)
    - Process stdin/stdout: Used for iteration control

Typical Workflow:
    1. Initialize SALT2mu object (launches subprocess)
    2. For each MCMC iteration:
       a. Prepare iteration (increment counter, open file)
       b. Write PDF functions for each parameter
       c. Close iteration (flush file)
       d. Send iteration number to subprocess
       e. Parse results from output file
"""

from __future__ import annotations

import logging
import subprocess
import time
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dust2dusty.logging import get_logger, setup_walker_logger

if TYPE_CHECKING:
    from dust2dusty.cli import Config

# Module-level logger
logger: logging.Logger = get_logger()


class SALT2mu:
    """
    Interface class for SALT2mu.exe subprocess communication.

    Manages persistent connection to SALT2mu.exe, handling PDF writing,
    subprocess control, and result parsing.

    Attributes:
        logger: Custom logger for this instance.
        iter: Current iteration number (starts at -1).
        debug: Debug mode flag.
        ready_enditer: Expected text indicating subprocess ready for next iteration.
        crosstalkfile: File handle for writing PDFs (mapsout).
        SALT2muoutputs: File handle for reading results (SALT2muout).
        command: Command string used to launch SALT2mu.exe.
        process: Subprocess object (if is_realdata=False).
        salt2mu_results: Dictionary containing parsed results from SALT2mu.
    """

    def __init__(
        self,
        command: str,
        mapsout: Path,
        salt2mu_out: Path,
        log_out: Path,
        is_realdata: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize SALT2mu connection.

        Args:
            command: Command string for SALT2mu.exe with %s placeholders for files.
                Format: "SALT2mu.exe input.file SUBPROCESS_FILES=%s,%s,%s ...".
            mapsout: Path for PDF crosstalk file (Python writes, SALT2mu reads).
            salt2mu_out: Path for results file (SALT2mu writes, Python reads).
            log: Path for subprocess log file.
            is_realdata: If True, run synchronously and return immediately.
            debug: If True, enable debug logging and YAML output.

        Side Effects:
            - If is_realdata=True: Runs SALT2mu via subprocess.run and calls getData()
            - If is_realdata=False: Launches SALT2mu.exe subprocess
        """
        # Get walker ID from mapsout filename for walker-specific logging
        walker_id = mapsout.name.split("_")[0]
        self.logger: logging.Logger = setup_walker_logger(walker_id, debug=debug)

        self.iter: int = -1
        self.debug: bool = debug
        self.ready_enditer: str = "Enter expected ITERATION number"
        self.done: str = "Graceful Program Exit. Bye."
        self.initready: str = "Finished SUBPROCESS_INIT"
        self.crosstalkfile = open(mapsout, "w")
        self.SALT2muoutputs = open(salt2mu_out)

        self.command: str = command % (
            mapsout.absolute(),
            salt2mu_out.absolute(),
            log_out.absolute(),
        )

        self.logger.info("Init SALT2mu instance. ")
        self.logger.info("## ================================== ##")
        self.logger.info(f"Command: {self.command}")
        self.logger.info(f"mapsout: {mapsout}")
        self.logger.debug("DEBUG MODE ON")

        self.logger.info("Command being run: " + self.command)
        self.salt2mu_results: dict[str, Any] = {}

        if is_realdata:
            if self.debug:
                self.command = self.command + " write_yaml=1"

            self.logger.info("Running realdata=True")
            self.logger.debug("## ==============RUN SALT2MU DATA================= ##")

            with (
                nullcontext()
                if self.debug
                else open(log_out.parent / f"{walker_id}_PROCESSLOG.log", "w")
            ) as stdout:
                subprocess.run(self.command, shell=True, stdout=stdout)
            self.getData()
            self.logger.debug("## =====================END======================= ##")

        else:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=0,
            )
            self.wait_until_text_in_output(self.ready_enditer)

    def quit(self) -> None:
        """
        Gracefully terminate the SALT2mu subprocess.

        Sends -1 to stdin which signals SALT2mu to exit, then reads
        remaining stdout until the process terminates.
        """
        self.process.stdin.write("-1\n")
        for stdout_line in iter(self.process.stdout.readline, ""):
            self.logger.info(stdout_line)

    def wait_until_text_in_output(self, expected_text: str, timeout: int = 120) -> None:
        """
        Wait for specific text to appear in subprocess stdout.

        Args:
            expected_text: Text string to search for in output.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Raises:
            TimeoutError: If expected_text not found within timeout period.
        """
        start = time.time()

        for line in iter(self.process.stdout.readline, ""):
            self.logger.debug(line.strip())
            if expected_text in line:
                self.logger.debug("found expected text")
                break

            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for '{expected_text}'")

    def next_iter(
        self,
        theta: NDArray[np.float64],
        theta_index_dic: dict[str, list[int]],
        config: Config,
    ) -> None:
        """
        Advance to next MCMC iteration.

        Writes PDF functions for all parameters based on current theta values,
        sends iteration number to subprocess, and parses results.

        Args:
            theta: Array of current parameter values.
            theta_index_dic: Mapping from parameter names to theta array indices.
            config: Configuration object with parameter specifications.
        """
        self.iter += 1
        self.write_iterbegin()

        for key in config.inp_params:
            key_parameter_values = theta[theta_index_dic[key][0] : theta_index_dic[key][-1] + 1]

            arr: list[NDArray[np.float64]] = []
            if key not in ["beta", "alpha"]:
                arr.append(config.DEFAULT_PARAMETER_RANGES[key])
                if key in config.splitdict.keys():
                    for s in config.splitdict[key].keys():
                        arr.append(eval(config.splitarr[s]))
            self.logger.debug(f"{key}")
            self.logger.debug(f"{config.paramshapesdict[key]}")
            self.logger.debug(f"{config.splitdict}")

            self.write_generic_PDF(
                key,
                config.splitdict,
                key_parameter_values,
                config.paramshapesdict[key],
                config.DISTRIBUTION_PARAMETERS,
                config.PARAM_TO_SALT2MU,
                arr,
            )

        self.logger.info("Launch next step")
        self.write_iterend()

        # Launch SALT2mu on new dist and wait for done
        self.process.stdin.write(f"{self.iter}\n")
        self.process.stdin.flush()
        self.wait_until_text_in_output(self.ready_enditer)

        self.data = self.getData()

    def getData(self) -> bool:
        """
        Parse SALT2mu output file to extract fit results.

        Reads the SALT2mu output file and extracts fit parameters and
        binned statistics into self.salt2mu_results dictionary.

        Returns:
            True upon successful parsing.

        Side Effects:
            Populates self.salt2mu_results with keys:
                - alpha, alphaerr: SALT2 standardization parameter and error
                - beta, betaerr: Color-luminosity parameter and error
                - maxprob: Maximum probability ratio (for boundary checking)
                - sigint, siginterr: Intrinsic scatter and error
                - bindf: pandas DataFrame with binned statistics
                - headerinfo: Tuple of (column_names, start_row)
        """
        self.SALT2muoutputs.seek(0)
        text = self.SALT2muoutputs.read()
        self.salt2mu_results["alpha"] = float(text.split("alpha0")[1].split()[1])
        self.salt2mu_results["alphaerr"] = float(text.split("alpha0")[1].split()[3])
        self.salt2mu_results["beta"] = float(text.split("beta0")[1].split()[1])
        self.salt2mu_results["betaerr"] = float(text.split("beta0")[1].split()[3])
        self.salt2mu_results["maxprob"] = float(text.split("MAXPROB_RATIO")[1].split()[1])
        self.salt2mu_results["headerinfo"] = self.NAndR(StringIO(text))
        self.salt2mu_results["sigint"] = float(text.split("sigint")[1].split()[1])
        self.salt2mu_results["bindf"] = pd.read_csv(
            StringIO(text),
            header=None,
            skiprows=self.salt2mu_results["headerinfo"][1],
            names=self.salt2mu_results["headerinfo"][0],
            delim_whitespace=True,
            comment="#",
        )
        self.salt2mu_results["siginterr"] = 0.0036  # DEFAULT VALUE
        return True

    def get_1d_asym_gauss(
        self,
        mean: float,
        lhs: float,
        rhs: float,
        arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate asymmetric Gaussian probability distribution.

        Creates a Gaussian with different widths left and right of the mean.

        Args:
            mean: Central value of the distribution.
            lhs: Standard deviation for values < mean.
            rhs: Standard deviation for values > mean.
            arr: Array of x values to evaluate PDF at.

        Returns:
            Tuple of (arr, probs) where probs are normalized to max=1.
        """
        probs = np.exp(-0.5 * ((arr - mean) / lhs) ** 2)
        probs[arr > mean] = np.exp(-0.5 * ((arr[arr > mean] - mean) / rhs) ** 2)
        probs = probs / np.max(probs)
        return arr, probs

    def get_1d_exponential(
        self, tau: float, arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate exponential probability distribution.

        Args:
            tau: Exponential decay constant (scale parameter).
            arr: Array of x values to evaluate PDF at (should be >= 0).

        Returns:
            Tuple of (arr, probs) where probs are normalized to max=1.
        """
        probs = (tau**-1) * np.exp(-arr / tau)
        probs = probs / np.max(probs)
        return arr, probs

    def get_1d_lognormal(
        self, mu: float, std: float, arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate log-normal probability distribution.

        Note:
            Current implementation uses exp(mu + std*arr) which may not be
            the standard log-normal formula. Review if unexpected behavior.

        Args:
            mu: Location parameter.
            std: Scale parameter.
            arr: Array of x values to evaluate PDF at.

        Returns:
            Tuple of (arr, probs) where probs are normalized to max=1.
        """
        probs = np.exp(mu + std * arr)
        probs = probs / np.max(probs)
        return arr, probs

    def writeheader(self, names: list[str]) -> None:
        """
        Write VARNAMES header line to crosstalk file.

        Deprecated: Use writegenericheader instead.

        Args:
            names: List of variable names.
        """
        self.crosstalkfile.write("VARNAMES:")
        for name in names:
            self.crosstalkfile.write(" " + name)
        self.crosstalkfile.write(" PROB\n")

    def writegenericheader(self, inp: str, varnames: list[str]) -> None:
        """
        Write VARNAMES header line for PDF block.

        Args:
            inp: Main variable name (e.g., 'SIM_c', 'SIM_RV').
            varnames: List of additional variable names for splits
                (e.g., ['HOST_LOGMASS'] for 2D PDF).

        Side Effects:
            Writes "VARNAMES: inp varname1 varname2 ... PROB\\n" to crosstalk file.
        """
        self.crosstalkfile.write(f"VARNAMES: {inp}")
        for name in varnames:
            self.crosstalkfile.write(" " + name)
        self.crosstalkfile.write(" PROB \n")

    def write3Dprobs(
        self,
        arr: NDArray[np.float64],
        z: float,
        mass: float,
        probs: NDArray[np.float64],
    ) -> None:
        """
        Write 3D probability distribution to crosstalk file.

        Args:
            arr: Array of primary variable values.
            z: Redshift value (scalar).
            mass: Mass value (scalar).
            probs: Array of probability values (same length as arr).

        Side Effects:
            Writes "PDF: value z mass prob" lines to crosstalk file.
        """
        bigstr = ""
        for a, p in zip(arr, probs):
            bigstr += "PDF: %.3f %.2f %.2f %.3f\n" % (a, z, mass, p)
        self.crosstalkfile.write(bigstr)

    def write2Dprobs(
        self, arr: NDArray[np.float64], mass: float, probs: NDArray[np.float64]
    ) -> None:
        """
        Write 2D probability distribution to crosstalk file.

        Args:
            arr: Array of primary variable values.
            mass: Mass value (scalar).
            probs: Array of probability values (same length as arr).

        Side Effects:
            Writes "PDF: value mass prob" lines to crosstalk file.
        """
        bigstr = ""
        for a, p in zip(arr, probs):
            bigstr += "PDF: %.3f %.2f %.3f\n" % (a, mass, p)
        self.crosstalkfile.write(bigstr)

    def write1Dprobs(self, arr: NDArray[np.float64], probs: NDArray[np.float64]) -> None:
        """
        Write 1D probability distribution to crosstalk file.

        Args:
            arr: Array of variable values.
            probs: Array of probability values (same length as arr).

        Side Effects:
            Writes "PDF: value prob" lines to crosstalk file with trailing newline.
        """
        bigstr = ""
        for a, p in zip(arr, probs):
            bigstr += "PDF: %.3f %.3f\n" % (a, p)
        bigstr += "\n"
        self.crosstalkfile.write(bigstr)

    def write_iterbegin(self) -> None:
        """
        Start new iteration in crosstalk file.

        Side Effects:
            - Truncates crosstalk file to zero length
            - Writes "ITERATION_BEGIN: N" where N = self.iter
        """
        self.crosstalkfile.truncate(0)
        self.crosstalkfile.seek(0)
        self.crosstalkfile.write("ITERATION_BEGIN: %d\n" % self.iter)

    def write_iterend(self) -> None:
        """
        Mark end of iteration in crosstalk file.

        Side Effects:
            - Writes "ITERATION_END: N" where N = self.iter
            - Flushes crosstalk file to ensure SALT2mu can read it
        """
        self.crosstalkfile.write("ITERATION_END: %d\n" % self.iter)
        self.crosstalkfile.flush()

    def write_SALT2(self, name: str, params: list[float]) -> None:
        """
        Write SALT2 standardization parameters (alpha/beta) in SNANA format.

        Alpha and beta are handled differently from other parameters - they use
        SNANA GENPEAK/GENSIGMA/GENRANGE format instead of PDF format.

        Args:
            name: Parameter name ('alpha' or 'beta').
            params: List [mean, std] for Gaussian distribution.

        Side Effects:
            Writes SNANA-format parameter specification to crosstalk file.
            GENRANGE is .1-.2 for alpha, .4-3 for beta.
        """
        for _ in range(3):
            self.crosstalkfile.write("\n")
        mean = params[0]
        std = params[1]
        self.crosstalkfile.write(f"GENPEAK_SIM_{name}: {mean} \n")
        self.crosstalkfile.write(f"GENSIGMA_SIM_{name}: {std} {std} \n")
        if name == "alpha":
            self.crosstalkfile.write(f"GENRANGE_SIM_{name}: .1 .2 \n")
        else:
            self.crosstalkfile.write(f"GENRANGE_SIM_{name}: .4 3 \n")
        for _ in range(3):
            self.crosstalkfile.write("\n")

    def write_1D_PDF(self, varname: str, params: list[float], arr: NDArray[np.float64]) -> None:
        """
        Write 1D PDF to crosstalk file.

        Deprecated: Use write_generic_PDF instead.

        Args:
            varname: Variable name.
            params: Distribution parameters [mean, lhs, rhs] or [mean, std].
            arr: Array of x values.
        """
        self.writeheader([varname])
        try:
            mean, lhs, rhs = params
        except ValueError:
            mean = params[0]
            lhs = params[1]
            rhs = lhs
        arr, probs = self.get_1d_asym_gauss(mean, lhs, rhs, arr)
        self.write1Dprobs(arr, probs)

    def NAndR(self, fp: StringIO) -> tuple[list[str], int]:
        """
        Parse SALT2mu output to find variable names and data start row.

        Searches for VARNAMES line to get column headers and identifies
        where actual data rows begin.

        Args:
            fp: File-like object (StringIO) with SALT2mu output text.

        Returns:
            Tuple of (Names, Startrow) where:
                - Names: List of column names
                - Startrow: Integer row number where data begins
        """
        Names: list[str] = []
        Startrow: int = 0
        for i, line in enumerate(fp):
            if line.startswith("VARNAMES:"):
                line = line.replace(",", " ")
                line = line.replace("\n", "")
                Names = line.split()
            elif line.startswith("SN") or line.startswith("ROW:") or line.startswith("GAL"):
                Startrow = i
                break
        return Names, Startrow

    def write_generic_PDF(
        self,
        inp: str,
        split: dict[str, dict[str, float]],
        params: NDArray[np.float64],
        shape: str,
        shapedict: dict[str, list[str]],
        simdict: dict[str, str],
        arr: list[NDArray[np.float64]],
    ) -> str | None:
        """
        Write probability distribution function for any parameter with arbitrary splits.

        Main PDF writing function that handles:
        - 1D PDFs (no splits)
        - 2D PDFs (one split variable, e.g., mass)
        - 3D PDFs (two split variables, e.g., redshift and mass)

        Args:
            inp: Parameter name (e.g., 'c', 'RV', 'EBV').
            split: Dictionary defining splits for this parameter.
                Format: {param: {split_var: split_value}}.
            params: Array of distribution parameters from MCMC.
            shape: Distribution shape name ('Gaussian', 'Exponential', etc.).
            shapedict: Dictionary mapping shapes to parameter names.
            simdict: Dictionary mapping parameter names to SALT2mu format.
            arr: List of arrays [param_values, split1_values, split2_values, ...].

        Returns:
            'Done' for alpha/beta early return, None otherwise.

        Side Effects:
            - Writes header and PDF data to crosstalk file
            - For RV: sets prob=0 for values < 0.4
        """
        if ("alpha" in inp) or ("beta" in inp):
            self.write_SALT2(inp, params)
            return "Done"

        splits: list[str] = []
        if inp in split.keys():
            splits = list(split[inp].keys())
        splits = [simdict[i] for i in splits]
        splitloc = [split[inp][i] for i in splits]
        self.writegenericheader(simdict[inp], splits)

        if len(splits) == 0:
            arrs, probs = self.shape_assigner(params, shape, arr[0])
            if inp == "RV":
                probs[arrs < 0.4] = 0
            self.write1Dprobs(arrs, probs)
        elif len(splits) == 1:
            splitparams = self.shape_interpret(params, inp, shape, shapedict, split)
            for sp1 in arr[1]:
                if sp1 < splitloc[0]:
                    arrs, probs = self.shape_assigner(splitparams[0], shape, arr[0])
                    if inp == "RV":
                        probs[arrs < 0.4] = 0
                    self.write2Dprobs(arrs, sp1, probs)
                else:
                    arrs, probs = self.shape_assigner(splitparams[1], shape, arr[0])
                    if inp == "RV":
                        probs[arrs < 0.4] = 0
                    self.write2Dprobs(arrs, sp1, probs)
        elif len(splits) == 2:
            splitparams = self.shape_interpret(params, inp, shape, shapedict, split)
            for sp1 in arr[1]:
                if np.around(sp1, 3) < splitloc[0]:
                    for sp2 in arr[2]:
                        if np.around(sp2, 1) < splitloc[1]:
                            arrs, probs = self.shape_assigner(splitparams[0], shape, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
                        else:
                            arrs, probs = self.shape_assigner(splitparams[1], shape, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
                else:
                    for sp2 in arr[2]:
                        if np.around(sp2, 1) < splitloc[1]:
                            arrs, probs = self.shape_assigner(splitparams[2], shape, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
                        else:
                            arrs, probs = self.shape_assigner(splitparams[3], shape, arr[0])
                            if inp == "RV":
                                probs[arrs < 0.4] = 0
                            self.write3Dprobs(arrs, sp1, sp2, probs)
        self.crosstalkfile.write("\n")
        return None

    def shape_assigner(
        self,
        params: NDArray[np.float64],
        shape: str,
        arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate probability distribution based on shape and parameters.

        Dispatcher function that calls appropriate distribution generator
        based on shape specification.

        Args:
            params: Distribution parameters (length depends on shape).
                - Gaussian: [mu, std]
                - Exponential: [tau]
                - LogNormal: [ln_mu, ln_std]
                - Skewed Gaussian: [mu, std_left, std_right]
            shape: Distribution shape name.
            arr: Array of x values to evaluate PDF at.

        Returns:
            Tuple of (arr, probs) from appropriate distribution function.
        """
        if shape == "Gaussian":
            return self.get_1d_asym_gauss(params[0], params[1], params[1], arr)
        elif shape == "Exponential":
            return self.get_1d_exponential(params[0], arr)
        elif shape == "LogNormal":
            return self.get_1d_lognormal(params[0], params[1], arr)
        elif shape == "Skewed Gaussian":
            return self.get_1d_asym_gauss(params[0], params[1], params[2], arr)
        else:
            raise ValueError(f"Unknown shape: {shape}")

    def shape_interpret(
        self,
        params: NDArray[np.float64],
        inp: str,
        shape: str,
        shapedict: dict[str, list[str]],
        split: dict[str, dict[str, float]],
    ) -> list[NDArray[np.float64]]:
        """
        Split params array into sub-arrays for each split combination.

        For parameters with splits, params contains parameters for all
        split combinations concatenated. This function splits them back out.

        Example:
            RV with mass split, Gaussian shape:
            params = [mu_low, std_low, mu_high, std_high]
            Returns: [[mu_low, std_low], [mu_high, std_high]]

        Args:
            params: Concatenated array of all distribution parameters.
            inp: Parameter name.
            shape: Distribution shape name.
            shapedict: Dictionary mapping shapes to parameter names.
            split: Dictionary defining splits for this parameter.

        Returns:
            List of parameter arrays, one per split combination.
            Length = 2^(number of split variables).
        """
        temp: list[NDArray[np.float64]] = []
        n_shape_params = len(shapedict[shape])
        for i in range(len(split[inp].keys()) * 2):
            temp.append(params[i * n_shape_params : (i + 1) * n_shape_params])
        return temp
