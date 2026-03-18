"""
MCMC sampling module for DUST2DUSTY.

This module contains the main MCMC sampling function supporting emcee
and nautilus samplers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import emcee
import h5py
import numpy as np
import schwimmbad
from numpy.typing import NDArray

from dust2dusty.likelihood_worker import (
    _init_worker,
    cleanup_worker,
    log_likelihood,
    log_probability,
)
from dust2dusty.log import get_logger, setup_logging
from dust2dusty.utils import get_sampled_par_names_and_init, write_chain_to_text

if TYPE_CHECKING:
    from dust2dusty.cli import Config

logger: logging.Logger = get_logger()


def MCMC(
    config: Config | None,
    realdata_salt2mu_results: dict[str, Any] | None,
    debug: bool = False,
    max_iterations: int = 100_000,
    convergence_check_interval: int = 100,
) -> None:
    """
    Run MCMC sampling using emcee or nautilus.

    For emcee: uses an HDF5 backend for robust chain storage and monitors
    convergence via integrated autocorrelation time.

    For nautilus: uses importance sampling with neural networks. Builds a
    uniform Prior from parameter bounds in config.parameter_initialization.
    Does not use MPI workers via schwimmbad; instead passes pool= to the
    nautilus Sampler directly (MPIPoolExecutor for MPI, None for serial).
    Chains are saved to the same HDF5 file via nautilus's filepath= argument.

    For MPI runs with emcee, worker processes (rank > 0) call this function
    with None values and wait in the pool for tasks from the master.
    Nautilus MPI workers are managed internally by mpi4py.futures and do not
    enter this code path.

    Args:
        config: Configuration object with parameters and paths (None for MPI workers).
        pos: Initial walker positions array of shape (nwalkers, ndim) (None for workers
            and unused by nautilus).
        nwalkers: Number of MCMC walkers (0 for workers, unused by nautilus).
        ndim: Number of parameters (dimensions) (0 for workers).
        realdata_salt2mu_results: Dictionary containing real data fit results (None for workers).
        debug: If True, run in debug mode (3 iterations for emcee, 3 likelihood calls for
            nautilus; no persistent HDF5 backend).
        debug_logging: If True, enable DEBUG-level logging on workers
            without changing MCMC execution or SALT2mu behavior.
        max_iterations: Maximum number of iterations / likelihood calls before stopping.
            For nautilus this maps to n_like_max in sampler.run().
        convergence_check_interval: Check convergence every N steps (emcee only).
        sampler: Which sampler to use: 'emcee' (default) or 'nautilus'.

    Returns:
        None.

    Convergence Criteria:
        emcee:
            1. Chain length > 100 * tau (autocorrelation time)
            2. Tau estimate changed by < 1% since last check
        nautilus:
            f_live < 0.01 and n_eff >= 10000 (nautilus defaults)

    Side Effects:
        - Saves chains to HDF5 file: {outdir}/chains/{data_input}-chains.h5
        - Saves autocorrelation history to: {outdir}/chains/{data_input}-autocorr.npz
          (emcee only)
        - Saves thinned/posterior samples to: {outdir}/chains/{data_input}-samples_thinned.npz
    """
    # Detect worker vs master before accessing config attributes.
    # Nautilus manages its own MPI workers via MPIPoolExecutor; schwimmbad
    # worker-wait only applies to emcee/zeus.
    is_worker = config is None

    # Validate sampler choice (only on master where config is available)
    if not is_worker and config.SAMPLER not in ("emcee", "nautilus"):
        raise ValueError(f"Unknown sampler '{config.SAMPLER}'. Choose 'emcee' or 'nautilus'.")

    if is_worker or config.USE_MPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    # WORKERS #
    if is_worker:
        # Set up basic logging for this worker process
        setup_logging(debug=False)
        # Receive initialization data from master
        worker_config, worker_realdata, likelihood_parameter, worker_debug = comm.bcast(
            None, root=0
        )
        sys.stdout.flush()
        _init_worker(worker_config, worker_realdata, likelihood_parameter, worker_debug)
        # Now enter the pool and wait for tasks
        with schwimmbad.MPIPool() as pool:
            pool.wait()
        sys.exit(0)

    # MASTER #
    par_names, p0_mu, p0_std, par_bounds, log_sampling = get_sampled_par_names_and_init(config)
    ndim = len(par_names)

    likelihood_parameters = {
        "par_names": par_names,
        "par_bounds": par_bounds,
        "log_sampling": log_sampling,
    }

    # Master send instruction to worker for init
    if config.USE_MPI:
        n_proc = comm.Get_size()
        # Broadcast initialization data to all workers BEFORE creating pool
        comm.bcast((config, realdata_salt2mu_results, likelihood_parameters, debug), root=0)
        pool = schwimmbad.MPIPool()
    # Not using MPI
    else:
        pool = schwimmbad.SerialPool()
        _init_worker(config, realdata_salt2mu_results, likelihood_parameters, debug)
        n_proc = 1
    # 2 walkers per worker is optimum
    emcee_nwalkers = int(2 * (n_proc - 1)) if n_proc > 1 else max(2 * ndim, 8)

    # Build the chain file (used by both samplers)
    chain_file = Path(config.OUTPUT_DIR) / "chains"
    chain_file /= config.data_input.stem + "-chains.h5"

    # Show log info
    logger.info("=" * 60)
    logger.info(f"Starting MCMC sampling ({config.SAMPLER})...")
    logger.info(f"  Dimensions: {ndim}")
    logger.info(f"  Parameters: {', '.join(par_names)}")
    if config.SAMPLER == "emcee":
        logger.info(f"  Walkers: {emcee_nwalkers}")
    if not debug:
        logger.info(f"  Chain file: {chain_file}")
    logger.debug("DEBUG MODE ON")
    logger.info("=" * 60 + "\n")

    with pool:
        # ------------------------------------------------------------------
        # Branch: nautilus (importance sampling + efficient space exploration using NN)
        # ------------------------------------------------------------------
        if config.SAMPLER == "nautilus":
            _run_nautilus(
                config=config,
                par_names=par_names,
                par_bounds=par_bounds,
                pool=pool,
                debug=debug,
                chain_file=chain_file,
                max_iterations=max_iterations,
            )

        # ------------------------------------------------------------------
        # Branch: emcee (Ensemble MCMC)
        # ------------------------------------------------------------------
        elif config.SAMPLER == "emcee":
            p0 = np.random.normal(p0_mu, p0_std, size=(emcee_nwalkers, ndim))

            _run_emcee(
                config=config,
                par_names=par_names,
                p0=p0,
                nwalkers=emcee_nwalkers,
                ndim=ndim,
                pool=pool,
                debug=debug,
                chain_file=chain_file,
                max_iterations=max_iterations,
                convergence_check_interval=convergence_check_interval,
            )

        # Shutting down
        cleanup_workers(pool, n_proc, logger)
    return None


# ---------------------------------------------------------------------------
# emcee implementation
# ---------------------------------------------------------------------------


def _run_emcee(
    config,
    par_names,
    p0,
    nwalkers,
    ndim,
    pool,
    debug,
    chain_file,
    max_iterations,
    convergence_check_interval,
):
    """
    Run emcee ensemble MCMC sampling until convergence or max iterations.

    In debug mode, runs exactly 3 steps, writes the chain to a text file
    alongside ``chain_file``, logs completion, and returns early without
    writing an HDF5 backend or autocorrelation history.

    In normal mode, runs the sampler in a loop and checks convergence every
    ``convergence_check_interval`` steps using the integrated autocorrelation
    time ``tau``.  Convergence is declared when both conditions hold:

        - chain length > 100 * tau  (sufficient independent samples)
        - relative change in tau since last check < 1%

    The loop runs for at most ``max_iterations`` steps. After the loop,
    autocorrelation history is saved as a ``.npz`` file and
    ``_finalize_emcee`` is called to thin and save the posterior samples.

    Args:
        config: Configuration object with parameter metadata and paths.
        pos: Initial walker positions, shape ``(nwalkers, ndim)``.
        nwalkers: Number of ensemble walkers.
        ndim: Number of parameters (dimensions).
        pool: schwimmbad pool (SerialPool or MPIPool) used by emcee.
        n_proc: Total number of MPI processes (1 for serial runs).
        debug: If True, run 3 steps only and save a debug text file.
        chain_file: ``pathlib.Path`` to the HDF5 chain file.
        max_iterations: Maximum number of sampler steps before stopping.
        convergence_check_interval: Check convergence every this many steps.

    Returns:
        None.
    """
    # Create backend file to save chain progress
    if debug:
        backend = None
    else:
        backend = emcee.backends.HDFBackend(chain_file, thin=5)
        backend.reset(nwalkers, ndim)

        with h5py.File(backend.filename, "a") as f:
            f[backend.name].attrs["parameter_names"] = par_names

        logger.debug(f"Chain storage initialized: {chain_file}")

        # Track autocorrelation time history
        autocorr_history = np.empty(max_iterations // convergence_check_interval)
        autocorr_index = 0
        old_tau: float | NDArray = np.inf

    # Init sampler
    sampler_obj = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        pool=pool,
        parameter_names=par_names,
        backend=backend,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ],
    )

    if debug:
        sampler_obj.run_mcmc(p0, 3)

        debug_chain_file = chain_file.with_name(chain_file.stem + "-debug_chains").with_suffix(
            ".txt"
        )

        write_chain_to_text(
            sampler_obj.get_chain(),
            sampler_obj.get_log_prob(),
            list(sampler_obj.parameter_names.keys()),
            debug_chain_file,
        )
        logger.info(f"Debug chains saved to: {debug_chain_file}")
        logger.info("DEBUG RUN COMPLETE (emcee, 3 steps).")
        return

    # Run with convergence monitoring
    for _ in sampler_obj.sample(p0, iterations=max_iterations, progress=False):
        if sampler_obj.iteration % convergence_check_interval:
            continue

        try:
            tau = sampler_obj.get_autocorr_time(tol=0)
            autocorr_history[autocorr_index] = np.mean(tau)
            autocorr_index += 1

            converged = np.all(tau * 100 < sampler_obj.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

            logger.info(f"\nIteration {sampler_obj.iteration}:")
            logger.info(f"  Mean tau: {np.mean(tau):.1f}")
            logger.info(f"  Min tau:  {np.min(tau):.1f}")
            logger.info(f"  Max tau:  {np.max(tau):.1f}")
            logger.info(
                f"  Chain/tau ratio: {sampler_obj.iteration / np.max(tau):.1f} (need > 100)"
            )
            if isinstance(old_tau, np.ndarray) and np.isfinite(old_tau).all():
                tau_change = np.max(np.abs(old_tau - tau) / tau) * 100
                logger.info(f"  Tau change: {tau_change:.2f}% (need < 1%)")

            if converged:
                logger.info("\n" + "=" * 60)
                logger.info("CONVERGENCE ACHIEVED!")
                logger.info(f"  Final iteration: {sampler_obj.iteration}")
                logger.info(f"  Final mean tau: {np.mean(tau):.1f}")
                logger.info("=" * 60)
                break

            old_tau = tau

        except emcee.autocorr.AutocorrError:
            logger.debug(f"\nIteration {sampler_obj.iteration}: Chain too short for tau estimate")

    # Save autocorrelation history
    autocorr_filename = chain_file.with_name(chain_file.stem + "-autocorr").with_suffix(".npz")
    np.savez(autocorr_filename, autocorr=autocorr_history[:autocorr_index])
    logger.info(f"Autocorrelation history saved to: {autocorr_filename}")

    _finalize_emcee(sampler_obj, chain_file, nwalkers)
    return None


def _finalize_emcee(
    sampler_obj,
    chain_file,
    nwalkers,
):
    """
    Compute burn-in and thinning from autocorrelation time and save thinned samples.

    Calls ``sampler_obj.get_autocorr_time()`` to obtain the final per-parameter
    autocorrelation time ``tau``.  Burn-in is set to ``2 * max(tau)`` and
    thinning to ``0.5 * min(tau)``.  Logs the effective sample count estimate
    and saves the thinned flat chain together with ``tau``, ``burnin``, and
    ``thin`` to a ``.npz`` file derived from ``chain_file``.

    If the chain is too short for a reliable autocorrelation estimate,
    ``emcee.autocorr.AutocorrError`` is caught and a warning is logged
    instead of raising.

    Args:
        config: Configuration object (currently unused but reserved for
            future parameter-name annotations).
        sampler_obj: Completed ``emcee.EnsembleSampler`` instance.
        chain_file: ``pathlib.Path`` to the HDF5 chain file; used to derive
            the output ``.npz`` filename.
        nwalkers: Number of ensemble walkers, used to estimate effective
            sample count.

    Returns:
        None.
    """
    logger.info("\n" + "=" * 60)
    logger.info("MCMC COMPLETE")
    logger.info("=" * 60)
    try:
        tau = sampler_obj.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        logger.info(f"Final autocorrelation time: {tau}")
        logger.info(f"Recommended burn-in: {burnin} steps")
        logger.info(f"Recommended thinning: {thin} steps")
        logger.info(f"Effective samples: ~{sampler_obj.iteration * nwalkers / np.mean(tau):.0f}")

        flat_samples = sampler_obj.get_chain(discard=burnin, thin=thin, flat=True)
        logger.info(f"Shape of thinned samples: {flat_samples.shape}")

        thinned_filename = chain_file.with_name(chain_file.stem + "-samples_thinned").with_suffix(
            ".npz"
        )
        np.savez(thinned_filename, samples=flat_samples, tau=tau, burnin=burnin, thin=thin)
        logger.info(f"Thinned samples saved to: {thinned_filename}")

    except emcee.autocorr.AutocorrError:
        logger.warning("Could not compute final autocorrelation time.")
        logger.warning("Chain may be too short for reliable estimates.")
        logger.warning("Consider running longer or checking for convergence issues.")
    return None


# ---------------------------------------------------------------------------
# nautilus implementation
# ---------------------------------------------------------------------------


def _run_nautilus(
    config,
    par_names,
    par_bounds,
    pool,
    debug,
    chain_file,
    max_iterations,
):
    """
    Run nautilus importance sampler until convergence or a likelihood-call cap.

    Builds a ``nautilus.Prior`` by iterating over the parameter names produced
    by ``pconv`` and adding each as a ``scipy.stats.uniform`` distribution
    whose support is taken from ``config.parameter_initialization[name]["bounds"]``.

    Creates a ``nautilus.Sampler`` with ``log_likelihood``, ``pass_dict=False``,
    and the provided ``pool``.  The HDF5 ``filepath`` and ``resume`` flags are
    set from ``chain_file`` unless in debug mode.

    Running behaviour:

        - Debug mode: caps likelihood evaluations at ``n_like_max=3``, logs
          completion, and returns without calling ``_finalize_nautilus``.
        - Normal mode: sets ``n_like_max=max_iterations`` and calls
          ``_finalize_nautilus`` after ``sampler_obj.run()`` completes.

    Args:
        config: Configuration object with parameter metadata and paths.
        realdata_salt2mu_results: Real-data SALT2mu fit results (currently
            forwarded for context; not directly consumed here).
        ndim: Number of parameters (dimensions).
        pool: Pool object passed directly to ``nautilus.Sampler``
            (MPIPoolExecutor for MPI, SerialPool for serial runs).
        n_proc: Total number of MPI processes (1 for serial runs).
        debug: If True, cap at 3 likelihood calls and skip finalisation.
        debug_logging: If True, DEBUG-level logging is active on workers
            (does not change sampling behaviour).
        chain_file: ``pathlib.Path`` to the HDF5 chain file passed to
            ``nautilus.Sampler`` as ``filepath``.
        max_iterations: Maximum number of likelihood calls (``n_like_max``)
            in normal mode.

    Returns:
        None.
    """
    try:
        from nautilus import Prior, Sampler
    except ImportError:
        logger.error("nautilus is not installed. Install it with: pip install nautilus-sampler")
        sys.exit(1)

    # Build Prior from parameter bounds
    from scipy.stats import uniform as scipy_uniform

    prior = Prior()
    for name, bounds in zip(par_names, par_bounds):
        lo, hi = bounds
        prior.add_parameter(name, dist=scipy_uniform(lo, hi - lo))

    logger.info(f"nautilus Prior built for {len(par_names)} parameters.")

    sampler_obj = Sampler(
        prior,
        log_likelihood,
        pass_dict=False,
        pool=pool,
        filepath=chain_file if not debug else None,
        resume=not debug,
    )

    logger.info(
        f"nautilus sampler created. Chain file: {chain_file if not debug else '(none, debug mode)'}"
    )

    run_kwargs: dict[str, Any] = {"verbose": True}
    if debug:
        # Minimal run: cap at 3 likelihood calls to match emcee/zeus debug behaviour
        run_kwargs["n_like_max"] = 3
    else:
        run_kwargs["n_like_max"] = max_iterations

    sampler_obj.run(**run_kwargs)

    if debug:
        logger.info("DEBUG RUN COMPLETE (nautilus, 3 likelihood calls).")
        return

    _finalize_nautilus(config, sampler_obj, chain_file)
    return


def _finalize_nautilus(config, sampler_obj, chain_file):
    """
    Extract equal-weight posterior samples from a completed nautilus sampler and save them.

    Calls ``sampler_obj.posterior(equal_weight=True)`` to obtain an
    unweighted draw from the posterior, which returns ``(points, log_w, log_l)``.
    Logs the number of posterior samples and the log evidence ``log_z``.
    Saves ``samples``, ``log_w``, ``log_l``, and ``log_z`` to a ``.npz``
    file derived from ``chain_file``.

    If posterior extraction fails for any reason, a warning is logged and
    the exception message is included, but no exception is re-raised.

    Args:
        config: Configuration object (currently unused but reserved for
            future parameter-name annotations).
        sampler_obj: Completed ``nautilus.Sampler`` instance.
        chain_file: ``pathlib.Path`` to the HDF5 chain file; used to derive
            the output ``.npz`` filename.

    Returns:
        None.
    """
    logger.info("\n" + "=" * 60)
    logger.info("MCMC COMPLETE (nautilus)")
    logger.info("=" * 60)

    try:
        # equal_weight=True returns an unweighted draw from the posterior
        points, log_w, log_l = sampler_obj.posterior(equal_weight=True)
        logger.info(f"Posterior samples: {points.shape[0]}")
        logger.info(f"log evidence (ln Z): {sampler_obj.log_z:.3f}")

        thinned_filename = chain_file.with_name(chain_file.stem + "-samples_thinned").with_suffix(
            ".npz"
        )
        np.savez(
            thinned_filename,
            samples=points,
            log_w=log_w,
            log_l=log_l,
            log_z=np.array([sampler_obj.log_z]),
        )
        logger.info(f"Posterior samples saved to: {thinned_filename}")

    except Exception as e:
        logger.warning(f"Could not extract posterior samples: {e}")
        logger.warning("Consider running longer or checking for convergence issues.")
    return


def cleanup_workers(pool, n_proc, logger):
    """
    Shut down all SALT2mu worker subprocesses.

    Maps ``cleanup_worker`` over ``range(n_proc)`` using ``pool.map`` so that
    every worker process (including MPI ranks) terminates its SALT2mu
    subprocess cleanly.  The result iterator is consumed immediately via
    ``list()`` to ensure all workers have finished before returning.

    Logs a message before dispatching the cleanup tasks and another after
    all workers have confirmed termination.

    Args:
        pool: schwimmbad pool (SerialPool or MPIPool) used to broadcast the
            cleanup call to every worker rank.
        n_proc: Total number of worker processes; determines the range of
            indices passed to ``cleanup_worker``.
        logger: Logger instance used to record start and completion messages.

    Returns:
        None.
    """
    logger.info("Shutting down SALT2mu subprocesses...")
    list(pool.map(cleanup_worker, range(n_proc)))
    logger.info("All SALT2mu subprocesses terminated.")
    return
