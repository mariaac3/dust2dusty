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
    - pdf_crosstalk_file: Python writes PDF functions here (input to SALT2mu)
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
import operator
import subprocess
import time
from contextlib import nullcontext
from io import StringIO
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dust2dusty.log import get_logger, setup_salt2mu_logger
from dust2dusty.utils import binned_dist

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
        pdf_crosstalk_file: File handle for writing PDFs (pdf_crosstalk_file).
        SALT2muoutputs: File handle for reading results (SALT2muout).
        command: Command string used to launch SALT2mu.exe.
        process: Subprocess object (if is_realdata=False).
        salt2mu_results: Dictionary containing parsed results from SALT2mu.
    """

    # Default value ranges for parameter arrays
    _DEFAULT_PARAMETER_GRID: ClassVar[dict[str, NDArray[np.float64]]] = {
        "c": np.arange(-0.5, 0.5, 0.001),
        "x1": np.arange(-5, 5, 0.01),
        "RV": np.arange(0, 8, 0.1),
        "EBV": np.arange(0.0, 1.5, 0.02),
        "EBVZ": np.arange(0.0, 1.5, 0.02),
    }

    _OPERATOR_MAP = {"low": operator.lt, "high": operator.gt}

    # Parameter name mappings for SALT2mu format
    _PARAM_TO_SALT2MU: ClassVar[dict[str, str]] = {
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
    }

    def __init__(
        self,
        command: str,
        genpdf_crosstalk_file: Path,
        salt2mu_out: Path,
        salt2mu_log_out: Path,
        is_realdata: bool = False,
        debug: bool = False,
        timeout=600,
        log_dir: Path | None = None,
        salt2mu_genpdf_grid: dict = {},
        split_dist_par: str | list = "HOST_LOGMASS",
    ) -> None:
        """
        Initialize SALT2mu connection.

        Args:
            command: Command string for SALT2mu.exe with %s placeholders for files.
                Format: "SALT2mu.exe input.file SUBPROCESS_FILES=%s,%s,%s ...".
            genpdf_crosstalk_file: Path for PDF crosstalk file (Python writes, SALT2mu reads).
            salt2mu_out: Path for results file (SALT2mu writes, Python reads).
            salt2mu_log_out: Path for subprocess log file.
            is_realdata: If True, run synchronously and return immediately.
            debug: If True, enable debug logging and YAML output.

        Side Effects:
            - If is_realdata=True: Runs SALT2mu via subprocess.run and calls getData()
            - If is_realdata=False: Launches SALT2mu.exe subprocess
        """
        # Get walker ID from pdf_crosstalk_file filename for walker-specific logging
        walker_id = genpdf_crosstalk_file.name.split("_")[0]

        if is_realdata:
            self.logger = logger
        else:
            self.logger: logging.Logger = setup_salt2mu_logger(
                walker_id, log_dir=log_dir, debug=debug
            )

        self.iter: int = 0
        self.timeout: int = timeout
        self.debug: bool = debug
        self.ready_enditer: str = "Enter expected ITERATION number"
        self.done: str = "Graceful Program Exit. Bye."
        self.initready: str = "Finished SUBPROCESS_INIT"

        # Init genpdf grid
        self.genpdf_grid_param_range = {
            **salt2mu_genpdf_grid,
            **self._DEFAULT_PARAMETER_GRID,
        }
        for key, val in self.genpdf_grid_param_range.items():
            if isinstance(val, dict):
                self.genpdf_grid_param_range[key] = getattr(np, val["method"])(*val["args"])
        self.split_split_par = split_dist_par
        self.genpdf_crosstalk_file = open(genpdf_crosstalk_file, "w")
        self.SALT2muoutputs = open(salt2mu_out)

        self.command: str = command % (
            genpdf_crosstalk_file.absolute(),
            salt2mu_out.absolute(),
            salt2mu_log_out.absolute(),
        )

        self.logger.info("===================== STARTS SALT2MU INSTANCE =====================\n")
        self.logger.info(f"  Command:\n {self.command} \n")
        self.logger.info(f"  GENPDF PYTHON CROSSTALK FILE: {genpdf_crosstalk_file}")
        self.logger.info(f"  SALT2MU RES FILE: {salt2mu_out}")
        self.logger.info(f"  SALT2MU LOG FILE: {salt2mu_log_out}")
        self.logger.debug("  WARNING => DEBUG MODE ON")

        self.salt2mu_results: dict[str, Any] = {}

        if is_realdata:
            self.iter = -9  # Default value for REAL DATA
            self.logger.info("\n ---------- RUN SALT2MU ON REAL DATA ---------- ##")

            with (
                nullcontext()
                if self.debug
                else open(salt2mu_log_out.parent / "REALDATA_SALT2MU_LOG.log", "w")
            ) as stdout:
                subprocess.run(self.command, shell=True, stdout=stdout)
            self.getData()
            self.logger.info(
                "\n =====================  END REAL DATA RUN ===================== \n\n"
            )

        else:
            self.logger.info("---------- INIT SALT2mu PROCESS ----------")
            start_time = time.time()
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.wait_until_text_in_output(self.ready_enditer)
            self.logger.info(
                f"\n## ===================== SALT2MU PROCESS INITIALIZED IN {time.time() - start_time:.0f} SECONDS ===================== ##\n\n"
            )

    def close(self) -> None:
        """
        Gracefully terminate the SALT2mu subprocess.

        Sends -1 to stdin which signals SALT2mu to exit, then reads
        remaining stdout until the process terminates.
        """
        self.process.stdin.write("-1\n")
        for stdout_line in self.process.stdout:
            self.logger.info(">>S2MU>> " + stdout_line)

    def wait_until_text_in_output(self, expected_text: str) -> None:
        """
        Wait for specific text to appear in subprocess stdout.

        Args:
            expected_text: Text string to search for in output.

        Raises:
            TimeoutError: If expected_text not found within timeout period.
        """
        start = time.time()

        for line in self.process.stdout:
            self.logger.debug(">>S2MU>> " + line.strip())
            if expected_text in line:
                self.logger.debug(f"FOUND '{expected_text}' => STOP WAITING")
                return

            if time.time() - start > self.timeout:
                raise TimeoutError(f"Timeout waiting for '{expected_text}'")

        self.logger.error(f"process poll -> {self.process.poll()}")
        self.logger.error(f"process code -> {self.process.returncode}")
        self.process.terminate()

        try:
            self.process.wait(timeout=5)  # Wait for the process to exit cleanly
        except subprocess.TimeoutExpired:
            self.process.kill()  # Force kill if it doesn't terminate

        outs, errs = self.process.communicate()  # Capture any remaining output after termination
        self.logger.error(f"Remaining output: {outs}")
        self.logger.error(f"Remaining errs: {errs}")

        raise RuntimeError("SALT2mu process terminated unexpectedly while waiting for output")

    def iterate(
        self,
        theta_dic: dict[str, float],
        fitted_par_dic: dict,
        last: bool = False,
    ) -> None:
        """
        Advance to next MCMC iteration.

        Writes PDF functions for all parameters based on current theta values,
        sends iteration number to subprocess, and parses results.

        Args:
            theta: Array of current parameter values.
            theta_index_dic: Mapping from parameter names to theta array indices.
            config: Configuration object with parameter specifications.
            last: If True, close the SALT2mu subprocess after this iteration.
        """
        if self.process.poll() is not None:
            return subprocess.SubprocessError("Subprocess has already terminated")
        self.write_iterbegin()

        ### Write PDF file with proposed "theta" parameters
        self.logger.debug(f"\n----- Write PDF file for iteration: {self.iter} -----\n")

        for param_name, param_dic in fitted_par_dic.items():
            param_dist_vals_dic = {k: v for k, v in theta_dic.items() if param_name in k}

            self.write_generic_PDF(
                param_name,
                param_dic,
                param_dist_vals_dic,
            )

        self.write_iterend()
        self.genpdf_crosstalk_file.flush()
        ### END WRITE PDF

        ### RUN SALT2mu on PDF file and wait for done
        self.logger.debug(f"\n----- Running SALT2mu iteration {self.iter} -----\n")

        self.process.stdin.write(f"{self.iter}\n")
        self.process.stdin.flush()
        self.wait_until_text_in_output(self.ready_enditer)  # wait SALT2mu execution
        ### END RUN SALT2mu

        ### Read the data
        self.data = self.getData()
        if last:
            self.close()

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
        """
        self.SALT2muoutputs.seek(0)
        lines = self.SALT2muoutputs.readlines()
        data_lines = []
        for l in lines:
            if "ITERATION" in l:
                iter = int(l.split(":")[-1])
                if int(l.split(":")[-1]) != self.iter:
                    raise RuntimeError(
                        f"SALT2mu output iteration does not match expected iteration. Seek {self.iter} found {iter}"
                    )
            if "+-" in l:
                parts = l.split("=")[1].split()
                if "alpha" in l:
                    k = "alpha"
                elif "beta" in l:
                    k = "beta"
                elif "sigint" in l:
                    k = "sigint"
                else:
                    continue
                self.salt2mu_results[k] = float(parts[0])
                self.salt2mu_results[k + "err"] = float(parts[2])
            elif "MAXPROB_RATIO" in l:
                self.salt2mu_results["maxprob"] = float(l.split()[4])
            elif l.startswith("VARNAMES"):
                names = l.split()[1:]
            elif l.startswith("ROW:"):
                data_lines.append(l.replace("ROW:", "").strip())
            else:
                pass

        self.salt2mu_results["binned_dists"] = binned_dist(
            pd.read_csv(
                StringIO("\n".join(data_lines)),
                names=names,
                sep=r"\s+",
                header=None,
            ),
            split_dist_par=self.split_split_par,
        )
        self.salt2mu_results["siginterr"] = 0.0036  # DEFAULT VALUE
        return True

    @staticmethod
    def get_1d_asym_gauss(
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
            Probability array normalized to max=1 (same shape as arr).
        """
        probs = np.exp(-0.5 * ((arr - mean) / lhs) ** 2)
        probs[arr > mean] = np.exp(-0.5 * ((arr[arr > mean] - mean) / rhs) ** 2)
        probs = probs / np.max(probs)
        return probs

    @staticmethod
    def get_1d_exponential(
        tau: float, arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate exponential probability distribution.

        Args:
            tau: Exponential decay constant (scale parameter).
            arr: Array of x values to evaluate PDF at (should be >= 0).

        Returns:
            Probability array normalized to max=1 (same shape as arr).
        """
        probs = (tau**-1) * np.exp(-arr / tau)
        probs = probs / np.max(probs)
        return probs

    @staticmethod
    def get_1d_lognormal(
        mu: float, std: float, arr: NDArray[np.float64]
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
            Probability array normalized to max=1 (same shape as arr).
        """
        probs = np.exp(mu + std * arr)
        probs = probs / np.max(probs)
        return probs

    def write_header(self, names: list[str]) -> None:
        """
        Write VARNAMES header line to pdf crosstalk file.

        Deprecated: Use write_generic_header instead.

        Args:
            names: List of variable names.
        """
        self.genpdf_crosstalk_file.write("VARNAMES:")
        for name in names:
            self.genpdf_crosstalk_file.write(" " + name)
        self.genpdf_crosstalk_file.write(" PROB\n")

    def write_generic_header(self, pdf_var_name: str, pdf_dep_var_names: list[str]) -> None:
        """
        Write VARNAMES header line for PDF block.

        Args:
            pdf_var_name: Main variable name (e.g., 'SIM_c', 'SIM_RV').
            pdf_dep_var_names: List of additional variable names for splits
                (e.g., ['HOST_LOGMASS'] for 2D PDF).

        Side Effects:
            Writes "VARNAMES: inp varname1 varname2 ... PROB\\n" to crosstalk file.
        """
        self.genpdf_crosstalk_file.write(
            f"VARNAMES: {pdf_var_name} " + " ".join(pdf_dep_var_names) + " PROB\n"
        )

    def write_GENPDF(
        self,
        genpdf_var_grid: NDArray[np.float64],
        genpdf_split_grid: list[NDArray[np.float64]],
        genpdf_probs_grid: NDArray[np.float64],
    ):
        """
        Write ND probability distribution to crosstalk file.

        Args:
            genpdf_var_grid: Meshgrid array of primary variable values.
            genpdf_split_grid: List of meshgrid arrays for split variables
                (empty list for 1D PDFs).
            genpdf_probs_grid: Meshgrid array of probability values
                (same shape as genpdf_var_grid).

        Side Effects:
            Writes "PDF: var_val [split_val ...] prob" lines to crosstalk file.
        """
        format_sring = "PDF: {:.3f}" + "{:8.2f}" * len(genpdf_split_grid) + "  {:8.3f}\n"
        bigstr = ""

        for a, *sp, p in zip(
            genpdf_var_grid.flatten(),
            *[sp.flatten() for sp in genpdf_split_grid],
            genpdf_probs_grid.flatten(),
        ):
            bigstr += format_sring.format(a, *sp, p)
        self.genpdf_crosstalk_file.write(bigstr)

    def write_iterbegin(self) -> None:
        """
        Start new iteration in crosstalk file.

        Side Effects:
            - Truncates crosstalk file to zero length
            - Writes "ITERATION_BEGIN: N" where N = self.iter
        """
        ### Clean PDF file
        self.genpdf_crosstalk_file.seek(0)
        self.genpdf_crosstalk_file.truncate()

        self.genpdf_crosstalk_file.write("ITERATION_BEGIN: %d\n" % self.iter)

    def write_iterend(self) -> None:
        """
        Mark end of iteration in crosstalk file.

        Side Effects:
            - Writes "ITERATION_END: N" where N = self.iter
            - Flushes crosstalk file to ensure SALT2mu can read it
        """
        self.genpdf_crosstalk_file.write("ITERATION_END: %d\n" % self.iter)

    def write_SALT2(self, name: str, param_dist_vals_dic: dict[str, float]) -> None:
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
        self.genpdf_crosstalk_file.write("\n" * 3)
        mean = param_dist_vals_dic[name + "_mu"]
        std = param_dist_vals_dic[name + "_std"]
        self.genpdf_crosstalk_file.write(f"GENPEAK_SIM_{name}: {mean} \n")
        self.genpdf_crosstalk_file.write(f"GENSIGMA_SIM_{name}: {std} {std} \n")
        if name == "alpha":
            self.genpdf_crosstalk_file.write(f"GENRANGE_SIM_{name}: .1 .2 \n")
        else:
            self.genpdf_crosstalk_file.write(f"GENRANGE_SIM_{name}: .4 3 \n")
        self.genpdf_crosstalk_file.write("\n" * 3)

    def write_generic_PDF(
        self,
        param_name: str,
        param_dic: dict,
        param_dist_vals_dic: NDArray[np.float64],
    ) -> str | None:
        """
        Write probability distribution function for any parameter with arbitrary splits.

        Main PDF writing function that handles:
        - 1D PDFs (no splits)
        - 2D PDFs (one split variable, e.g., mass)
        - 3D PDFs (two split variables, e.g., redshift and mass)

        Args:
            param_name: Internal parameter name (e.g., 'c', 'RV', 'EBV',
                'alpha', 'beta').
            param_dic: Per-parameter config dict from fitted_params (contains
                'dist' shape name and optional 'splits' dict).
            param_dist_vals_dic: Dict mapping expanded parameter names to
                current MCMC values (e.g., {'RV_mu_HOST_LOGMASS_low': 2.3, ...}).

        Returns:
            None (alpha/beta return early after writing SNANA-format lines).

        Side Effects:
            - Writes header and PDF data to crosstalk file
            - For RV: sets prob=0 for values < 0.4
        """
        if param_name in ["alpha", "beta"]:
            self.write_SALT2(param_name, param_dist_vals_dic)
            return

        par_name_SALT2MU = self._PARAM_TO_SALT2MU[param_name]

        # Handle splits
        split_pars = []
        if "splits" in param_dic:
            split_pars = list(param_dic["splits"].keys())
        split_par_SALT2MU = [self._PARAM_TO_SALT2MU[split_p] for split_p in split_pars]

        # Write header
        self.write_generic_header(par_name_SALT2MU, split_par_SALT2MU)

        # Init GENPDF distribution grid
        genpdf_par_grid, *genpdf_split_grid = np.meshgrid(
            self.genpdf_grid_param_range[param_name],
            *[self.genpdf_grid_param_range[split_p] for split_p in split_pars],
        )

        genpdf_split_grid_dic = {
            split_p: genpdf_split_g
            for split_p, genpdf_split_g in zip(split_pars, genpdf_split_grid)
        }
        genpdf_probs_grid = np.zeros_like(genpdf_par_grid)

        # all splits possible combinations
        split_combs = product(
            *[[f"{split_p}_{l}" for l in ["low", "high"]] for split_p in split_pars]
        )
        for comb in split_combs:
            key = "_" + "_".join(comb)

            # 1D case
            if key == "_":
                key = ""

            param_dist_vals_subdic = {
                k.replace(f"{key}", ""): v for k, v in param_dist_vals_dic.items() if key in k
            }

            mask = np.ones_like(genpdf_probs_grid, dtype=bool)
            for split_p in split_pars:
                if f"{split_p}_low" in key:
                    mask &= self._OPERATOR_MAP["low"](
                        genpdf_split_grid_dic[f"{split_p}"], param_dic["splits"][f"{split_p}"]
                    )
                elif f"{split_p}_high" in key:
                    mask &= self._OPERATOR_MAP["high"](
                        genpdf_split_grid_dic[f"{split_p}"], param_dic["splits"][f"{split_p}"]
                    )
                else:
                    raise ValueError(f"{split_p} key not found")

            genpdf_probs_grid[mask] = self.shape_assigner(
                param_name, param_dic["dist"], param_dist_vals_subdic, genpdf_par_grid[mask]
            )

        if "RV" in param_name:
            genpdf_probs_grid[genpdf_par_grid < 0.4] = 0

        self.write_GENPDF(genpdf_par_grid, genpdf_split_grid, genpdf_probs_grid)
        self.genpdf_crosstalk_file.write("\n")
        return None

    def shape_assigner(
        self,
        param_name: str,
        dist_shape: str,
        param_dist_vals_dic: NDArray[np.float64],
        genpdf_grid: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate probability distribution based on shape and parameters.

        Dispatcher function that calls appropriate distribution generator
        based on shape specification.

        Args:
            param_name: Internal parameter name used to look up keys in
                param_dist_vals_dic (e.g., 'RV', 'c').
            dist_shape: Distribution shape name — one of 'Gaussian',
                'Skewed Gaussian', 'Exponential', 'LogNormal'.
            param_dist_vals_dic: Dict of MCMC parameter values for this
                split/bin (e.g., {'RV_mu': 2.3, 'RV_std': 0.8}).
            genpdf_grid: Array of x values to evaluate the PDF at.

        Returns:
            Probability array (same shape as genpdf_grid), normalized to max=1.

        Raises:
            ValueError: If dist_shape is not a recognised distribution name.
        """
        if dist_shape == "Gaussian":
            return self.get_1d_asym_gauss(
                param_dist_vals_dic[param_name + "_mu"],
                param_dist_vals_dic[param_name + "_std"],
                param_dist_vals_dic[param_name + "_std"],
                genpdf_grid,
            )
        elif dist_shape == "Exponential":
            return self.get_1d_exponential(param_dist_vals_dic[param_name + "_tau"], genpdf_grid)
        elif dist_shape == "LogNormal":
            return self.get_1d_lognormal(
                param_dist_vals_dic[param_name + "_ln_mu"],
                param_dist_vals_dic[param_name + "_ln_std"],
                genpdf_grid,
            )
        elif dist_shape == "Skewed Gaussian":
            return self.get_1d_asym_gauss(
                param_dist_vals_dic[param_name + "_mu"],
                param_dist_vals_dic[param_name + "_std_low"],
                param_dist_vals_dic[param_name + "_std_high"],
                genpdf_grid,
            )
        else:
            raise ValueError(f"Unknown shape: {dist_shape}")
