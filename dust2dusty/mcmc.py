"""
MCMC sampling module for DUST2DUSTY.

This module contains the main MCMC sampling function using emcee.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np
import schwimmbad
from numpy.typing import NDArray

from dust2dusty.dust2dust import _init_worker, log_probability
from dust2dusty.logging import get_logger

if TYPE_CHECKING:
    from dust2dusty.cli import Config

logger: logging.Logger = get_logger()


def MCMC(
    config: Config | None,
    pos: NDArray[np.float64] | None,
    nwalkers: int,
    ndim: int,
    realdata_salt2mu_results: dict[str, Any] | None,
    debug: bool = False,
    max_iterations: int = 100000,
    convergence_check_interval: int = 100,
) -> emcee.EnsembleSampler | None:
    """
    Run MCMC sampling using emcee ensemble sampler with HDF5 backend.

    Uses the emcee HDF5 backend for robust chain storage and monitors
    convergence via integrated autocorrelation time. Sampling stops when
    chains are sufficiently long relative to autocorrelation time and tau
    estimates have stabilized.

    For MPI runs, worker processes (rank > 0) call this function with None
    values and wait in the pool for tasks from the master.

    Args:
        config: Configuration object with parameters and paths (None for MPI workers).
        pos: Initial walker positions array of shape (nwalkers, ndim) (None for workers).
        nwalkers: Number of MCMC walkers (0 for workers).
        ndim: Number of parameters (dimensions) (0 for workers).
        realdata_salt2mu_results: Dictionary containing real data fit results (None for workers).
        debug: If True, run in debug mode.
        max_iterations: Maximum number of iterations before stopping.
        convergence_check_interval: Check convergence every N steps.

    Returns:
        The emcee sampler object with chain results (master only).
        None for worker processes.

    Convergence Criteria (from emcee documentation):
        1. Chain length > 100 * tau (autocorrelation time)
        2. Tau estimate changed by < 1% since last check

    Side Effects:
        - Saves chains to HDF5 file: {outdir}/chains/{data_input}-chains.h5
        - Saves autocorrelation history to: {outdir}/chains/{data_input}-autocorr.npz
        - Saves thinned samples to: {outdir}/chains/{data_input}-samples_thinned.npz
    """
    # Check if this is a worker process (called with None config)
    is_worker = config is None

    if is_worker:
        # Worker process: receive initialization data via broadcast and enter pool
        with schwimmbad.MPIPool() as pool:
            # Receive initialization data from master
            worker_config, worker_realdata, worker_debug = pool.comm.bcast(None, root=0)
            _init_worker(worker_config, worker_realdata, worker_debug)
            pool.wait()
        sys.exit(0)

    # Master process continues with full MCMC setup
    # Set up HDF5 backend for robust chain storage
    chain_filename = (
        config.outdir + "chains/" + config.data_input.split(".")[0].split("/")[-1] + "-chains.h5"
    )
    backend = emcee.backends.HDFBackend(chain_filename)
    backend.reset(nwalkers, ndim)
    logger.debug(f"Chain storage initialized: {chain_filename}")

    # Track autocorrelation time history
    autocorr_history = np.empty(max_iterations // convergence_check_interval)
    autocorr_index = 0
    old_tau: float | NDArray = np.inf

    # Choose pool type - MPIPool doesn't support initializer argument
    if config.USE_MPI:
        pool = schwimmbad.MPIPool()
        n_proc = pool.comm.Get_size()
        # Broadcast initialization data to all workers
        pool.comm.bcast((config, realdata_salt2mu_results, debug), root=0)
    else:
        pool = schwimmbad.SerialPool()
        _init_worker(config, realdata_salt2mu_results, debug)
        n_proc = 1

    with pool:
        logger.info(
            f"Initializing MCMC with {n_proc} CPUs, {nwalkers} walkers, {ndim} dimensions, pool type is {pool.__class__.__name__}"
        )

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)

        logger.debug("=" * 60)
        if debug:
            sampler.run_mcmc(pos, 3)
            sys.exit(0)
        # Run with convergence monitoring
        for _ in sampler.sample(pos, iterations=max_iterations, progress=True):
            # Only check convergence every N steps
            if sampler.iteration % convergence_check_interval:
                continue

            # Compute autocorrelation time
            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr_history[autocorr_index] = np.mean(tau)
                autocorr_index += 1

                # Check convergence criteria
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

                logger.debug(f"\nIteration {sampler.iteration}:")
                logger.debug(f"  Mean tau: {np.mean(tau):.1f}")
                logger.debug(f"  Min tau:  {np.min(tau):.1f}")
                logger.debug(f"  Max tau:  {np.max(tau):.1f}")
                logger.debug(
                    f"  Chain/tau ratio: {sampler.iteration / np.max(tau):.1f} (need > 100)"
                )
                if isinstance(old_tau, np.ndarray) and np.isfinite(old_tau).all():
                    tau_change = np.max(np.abs(old_tau - tau) / tau) * 100
                    logger.debug(f"  Tau change: {tau_change:.2f}% (need < 1%)")

                if converged:
                    logger.debug("\n" + "=" * 60)
                    logger.debug("CONVERGENCE ACHIEVED!")
                    logger.debug(f"  Final iteration: {sampler.iteration}")
                    logger.debug(f"  Final mean tau: {np.mean(tau):.1f}")
                    logger.debug("=" * 60)
                    break

                old_tau = tau

            except emcee.autocorr.AutocorrError:
                logger.debug(f"\nIteration {sampler.iteration}: Chain too short for tau estimate")

        # Save autocorrelation history
        autocorr_filename = (
            config.outdir
            + "chains/"
            + config.data_input.split(".")[0].split("/")[-1]
            + "-autocorr.npz"
        )
        np.savez(autocorr_filename, autocorr=autocorr_history[:autocorr_index])
        logger.debug(f"Autocorrelation history saved to: {autocorr_filename}")

        # Report final statistics
        logger.debug("\n" + "=" * 60)
        logger.debug("MCMC COMPLETE")
        logger.debug("=" * 60)
        try:
            tau = sampler.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            logger.debug(f"Final autocorrelation time: {tau}")
            logger.debug(f"Recommended burn-in: {burnin} steps")
            logger.debug(f"Recommended thinning: {thin} steps")
            logger.debug(f"Effective samples: ~{sampler.iteration * nwalkers / np.mean(tau):.0f}")

            # Get flattened samples with burn-in and thinning applied
            flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
            logger.debug(f"Shape of thinned samples: {flat_samples.shape}")

            # Save thinned samples for convenience
            thinned_filename = (
                config.outdir
                + "chains/"
                + config.data_input.split(".")[0].split("/")[-1]
                + "-samples_thinned.npz"
            )
            np.savez(thinned_filename, samples=flat_samples, tau=tau, burnin=burnin, thin=thin)
            logger.debug(f"Thinned samples saved to: {thinned_filename}")

        except emcee.autocorr.AutocorrError:
            logger.warning("Could not compute final autocorrelation time.")
            logger.warning("Chain may be too short for reliable estimates.")
            logger.warning("Consider running longer or checking for convergence issues.")

    return sampler
