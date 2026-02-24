"""
MCMC sampling module for DUST2DUSTY.

This module contains the main MCMC sampling function supporting emcee,
zeus, and nautilus samplers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np
import schwimmbad
from numpy.typing import NDArray

from dust2dusty.dust2dust import _init_worker, cleanup_worker, log_likelihood, log_probability
from dust2dusty.log import get_logger, setup_logging
from dust2dusty.utils import pconv, write_chain_to_text

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
    debug_logging: bool = False,
    max_iterations: int = 100000,
    convergence_check_interval: int = 100,
    sampler: str = "emcee",
) -> None:
    """
    Run MCMC sampling using emcee, zeus, or nautilus.

    For emcee: uses an HDF5 backend for robust chain storage and monitors
    convergence via integrated autocorrelation time.

    For zeus: uses SaveProgressCallback for HDF5 chain storage and
    AutocorrelationCallback for convergence.

    For nautilus: uses importance sampling with neural networks. Builds a
    uniform Prior from parameter bounds in config.parameter_initialization.
    Does not use MPI workers via schwimmbad; instead passes pool= to the
    nautilus Sampler directly (MPIPoolExecutor for MPI, None for serial).
    Chains are saved to the same HDF5 file via nautilus's filepath= argument.

    For MPI runs with emcee/zeus, worker processes (rank > 0) call this
    function with None values and wait in the pool for tasks from the master.
    Nautilus MPI workers are managed internally by mpi4py.futures and do not
    enter this code path.

    Args:
        config: Configuration object with parameters and paths (None for MPI workers).
        pos: Initial walker positions array of shape (nwalkers, ndim) (None for workers
            and unused by nautilus).
        nwalkers: Number of MCMC walkers (0 for workers, unused by nautilus).
        ndim: Number of parameters (dimensions) (0 for workers).
        realdata_salt2mu_results: Dictionary containing real data fit results (None for workers).
        debug: If True, run in debug mode (3 iterations for emcee/zeus, 1 shell for
            nautilus; no persistent HDF5 backend; SALT2mu optmask=1).
        debug_logging: If True, enable DEBUG-level logging on workers
            without changing MCMC execution or SALT2mu behavior.
        max_iterations: Maximum number of iterations / likelihood calls before stopping.
            For nautilus this maps to n_like_max in sampler.run().
        convergence_check_interval: Check convergence every N steps (emcee/zeus only).
        sampler: Which sampler to use: 'emcee' (default), 'zeus', or 'nautilus'.

    Returns:
        The sampler object with chain results (master only).
        None for worker processes.

    Convergence Criteria:
        emcee:
            1. Chain length > 100 * tau (autocorrelation time)
            2. Tau estimate changed by < 1% since last check
        zeus:
            AutocorrelationCallback with ncheck=100, dact=0.01, nact=10
        nautilus:
            f_live < 0.01 and n_eff >= 10000 (nautilus defaults)

    Side Effects:
        - Saves chains to HDF5 file: {outdir}/chains/{data_input}-chains.h5
        - Saves autocorrelation history to: {outdir}/chains/{data_input}-autocorr.npz
          (emcee only)
        - Saves thinned/posterior samples to: {outdir}/chains/{data_input}-samples_thinned.npz
    """
    # Validate sampler choice first (before any MPI logic)
    if sampler not in ("emcee", "zeus", "nautilus"):
        raise ValueError(f"Unknown sampler '{sampler}'. Choose 'emcee', 'zeus', or 'nautilus'.")

    # Detect worker vs master before accessing config attributes.
    # Nautilus manages its own MPI workers via MPIPoolExecutor; schwimmbad
    # worker-wait only applies to emcee/zeus.
    is_worker = config is None

    if is_worker or config.USE_MPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    # WORKERS #
    if is_worker:
        # Set up basic logging for this worker process
        setup_logging(debug=False)
        # Receive initialization data from master
        worker_config, worker_realdata, worker_debug = comm.bcast(None, root=0)
        sys.stdout.flush()
        _init_worker(worker_config, worker_realdata, worker_debug)
        # Now enter the pool and wait for tasks
        with schwimmbad.MPIPool() as pool:
            pool.wait()
        sys.exit(0)

    # MASTER #

    # Build the chain file (used by both samplers)
    chain_file = Path(
        config.outdir + "chains/" + config.data_input.split(".")[0].split("/")[-1] + "-chains.h5"
    )
    # Show log info
    logger.info("=" * 60)
    logger.info(f"Starting MCMC sampling ({sampler})...")
    logger.info(f"  Dimensions: {ndim}")
    logger.info(f"  Parameters: {', '.join(config.inp_params)}")
    if not debug:
        logger.info(f"  Chain file: {chain_file}")
    if sampler in ("emcee", "zeus"):
        logger.info(f"  Walkers: {nwalkers}")
    logger.debug("DEBUG MODE ON")
    logger.info("=" * 60 + "\n")

    # Master send instruction to worker for init
    worker_debug = debug or debug_logging
    if config.USE_MPI:
        n_proc = comm.Get_size()
        # Broadcast initialization data to all workers BEFORE creating pool
        comm.bcast((config, realdata_salt2mu_results, worker_debug), root=0)
        pool = schwimmbad.MPIPool()
    # Not using MPI
    else:
        pool = schwimmbad.SerialPool()
        _init_worker(config, realdata_salt2mu_results, worker_debug)
        n_proc = 1

    with pool:
        # ------------------------------------------------------------------
        # Branch: nautilus
        # ------------------------------------------------------------------
        if sampler == "nautilus":
            _run_nautilus(
                config=config,
                realdata_salt2mu_results=realdata_salt2mu_results,
                ndim=ndim,
                pool=pool,
                n_proc=n_proc,
                debug=debug,
                debug_logging=debug_logging,
                chain_file=chain_file,
                max_iterations=max_iterations,
            )

        # ------------------------------------------------------------------
        # Branch: emcee
        # ------------------------------------------------------------------
        elif sampler == "emcee":
            _run_emcee(
                config=config,
                pos=pos,
                nwalkers=nwalkers,
                ndim=ndim,
                pool=pool,
                n_proc=n_proc,
                debug=debug,
                chain_file=chain_file,
                max_iterations=max_iterations,
                convergence_check_interval=convergence_check_interval,
                autocorr_history=autocorr_history if not debug else None,
                autocorr_index=autocorr_index if not debug else 0,
                old_tau=old_tau if not debug else np.inf,
            )

        # ------------------------------------------------------------------
        # Branch: zeus
        # ------------------------------------------------------------------
        elif sampler == "zeus":
            _run_zeus(
                config=config,
                pos=pos,
                nwalkers=nwalkers,
                ndim=ndim,
                pool=pool,
                n_proc=n_proc,
                debug=debug,
                chain_file=chain_file,
                max_iterations=max_iterations,
                convergence_check_interval=convergence_check_interval,
            )
        else:
            raise ValueError(f"Unknown sampler '{sampler}'. Choose 'emcee', 'zeus', or 'nautilus'.")

        # Shutting down
        cleanup_workers(pool, n_proc, logger)
    return None


# ---------------------------------------------------------------------------
# emcee implementation
# ---------------------------------------------------------------------------


def _run_emcee(
    config,
    pos,
    nwalkers,
    ndim,
    pool,
    n_proc,
    debug,
    chain_file,
    max_iterations,
    convergence_check_interval,
    autocorr_history,
    autocorr_index,
    old_tau,
):
    # Create backend file to save chain progress
    if debug:
        backend = None
    else:
        backend = emcee.backends.HDFBackend(chain_file)
        backend.reset(nwalkers, ndim)
        logger.debug(f"Chain storage initialized: {chain_file}")

        # Track autocorrelation time history
        autocorr_history = np.empty(max_iterations // convergence_check_interval)
        autocorr_index = 0
        old_tau: float | NDArray = np.inf

    # Init sampler
    sampler_obj = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)

    if debug:
        sampler_obj.run_mcmc(pos, 3)

        param_names = pconv(
            config.inp_params,
            config.paramshapesdict,
            config.splitdict,
            config.DISTRIBUTION_PARAMETERS,
        )
        debug_chain_file = chain_file.with_suffix("-debug_chains.txt")
        write_chain_to_text(
            sampler_obj.get_chain(),
            sampler_obj.get_log_prob(),
            param_names,
            debug_chain_file,
        )
        logger.info(f"Debug chains saved to: {debug_chain_file}")
        logger.info("DEBUG RUN COMPLETE (emcee, 3 steps).")
        return

    # Run with convergence monitoring
    for _ in sampler_obj.sample(pos, iterations=max_iterations, progress=False):
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
    autocorr_filename = config.outdir + "chains/" + chain_stem + "-autocorr.npz"
    np.savez(autocorr_filename, autocorr=autocorr_history[:autocorr_index])
    logger.info(f"Autocorrelation history saved to: {autocorr_filename}")

    _finalize_emcee(config, sampler_obj, chain_file, nwalkers)
    return None


def _finalize_emcee(
    config,
    sampler_obj,
    chain_file,
    nwalkers,
):
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

        thinned_filename = chain_file.with_suffix("-samples_thinned.npz")
        np.savez(thinned_filename, samples=flat_samples, tau=tau, burnin=burnin, thin=thin)
        logger.info(f"Thinned samples saved to: {thinned_filename}")

    except emcee.autocorr.AutocorrError:
        logger.warning("Could not compute final autocorrelation time.")
        logger.warning("Chain may be too short for reliable estimates.")
        logger.warning("Consider running longer or checking for convergence issues.")
    return None


# ---------------------------------------------------------------------------
# zeus implementation
# ---------------------------------------------------------------------------


def _run_zeus(
    config,
    pos,
    nwalkers,
    ndim,
    pool,
    n_proc,
    debug,
    chain_file,
    max_iterations,
    convergence_check_interval,
):
    try:
        import zeus
    except ImportError:
        logger.error("zeus is not installed. Install it with: pip install zeus-mcmc")
        sys.exit(1)

    sampler_obj = zeus.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, verbose=False)

    if debug:
        sampler_obj.run_mcmc(pos, 3)

        param_names = pconv(
            config.inp_params,
            config.paramshapesdict,
            config.splitdict,
            config.DISTRIBUTION_PARAMETERS,
        )
        debug_chain_file = chain_file.with_suffix("-debug_chains.txt")
        # zeus get_chain returns shape (nsteps, nwalkers, ndim)
        chain = sampler_obj.get_chain()
        log_prob = sampler_obj.get_log_prob()
        write_chain_to_text(chain, log_prob, param_names, debug_chain_file)
        logger.info(f"Debug chains saved to: {debug_chain_file}")
        logger.info("DEBUG RUN COMPLETE (zeus, 3 steps).")
        return

    # Build callbacks
    callbacks = [
        zeus.callbacks.AutocorrelationCallback(
            ncheck=convergence_check_interval,
            dact=0.01,
            nact=10,
            discard=0.5,
            trigger=True,
        ),
        zeus.callbacks.MinIterCallback(nmin=convergence_check_interval),
        zeus.callbacks.SaveProgressCallback(
            filename=chain_file,
            ncheck=convergence_check_interval,
        ),
    ]

    logger.info(
        f"zeus callbacks: AutocorrelationCallback (ncheck={convergence_check_interval}, "
        f"dact=0.01, nact=10), SaveProgressCallback -> {chain_file}"
    )

    sampler_obj.run_mcmc(pos, max_iterations, callbacks=callbacks, progress=True)

    _finalize_zeus(config, sampler_obj, chain_file)
    return


def _finalize_zeus(config, sampler_obj, chain_file):
    logger.info("\n" + "=" * 60)
    logger.info("MCMC COMPLETE (zeus)")
    logger.info("=" * 60)

    try:
        act = sampler_obj.act  # integrated autocorrelation time per parameter
        ess = sampler_obj.ess  # effective sample size
        logger.info(f"Integrated autocorrelation time: {act}")
        logger.info(f"Effective sample size: {ess}")
        logger.info(f"Efficiency: {sampler_obj.efficiency:.3f}")
        logger.info(f"Total log-prob calls: {sampler_obj.ncall}")

        burnin = int(2 * np.max(act))
        thin = max(1, int(0.5 * np.min(act)))
        flat_samples = sampler_obj.get_chain(discard=burnin, thin=thin, flat=True)
        logger.info(f"Recommended burn-in: {burnin} steps")
        logger.info(f"Recommended thinning: {thin} steps")
        logger.info(f"Shape of thinned samples: {flat_samples.shape}")

        thinned_file = chain_file.with_suffix("-samples_thinned.npz")
        np.savez(
            thinned_file,
            samples=flat_samples,
            act=act,
            burnin=burnin,
            thin=thin,
        )
        logger.info(f"Thinned samples saved to: {thinned_file}")

    except Exception as e:
        logger.warning(f"Could not compute final diagnostics: {e}")
        logger.warning("Consider running longer or checking for convergence issues.")
    return


# ---------------------------------------------------------------------------
# nautilus implementation
# ---------------------------------------------------------------------------


def _run_nautilus(
    config,
    realdata_salt2mu_results,
    ndim,
    pool,
    n_proc,
    debug,
    debug_logging,
    chain_file,
    max_iterations,
):
    try:
        from nautilus import Prior, Sampler
    except ImportError:
        logger.error("nautilus is not installed. Install it with: pip install nautilus-sampler")
        sys.exit(1)

    # Build Prior from parameter bounds
    param_names = pconv(
        config.inp_params,
        config.paramshapesdict,
        config.splitdict,
        config.DISTRIBUTION_PARAMETERS,
    )
    prior = Prior()
    for name in param_names:
        lo, hi = config.parameter_initialization[name]["bounds"]
        from scipy.stats import uniform as scipy_uniform

        prior.add_parameter(name, dist=scipy_uniform(lo, hi - lo))

    logger.info(f"nautilus Prior built for {len(param_names)} parameters.")

    sampler_obj = Sampler(
        prior,
        log_likelihood,
        pass_dict=False,
        pool=pool,
        filepath=chain_file if not debug else None,
        resume=not debug,
    )

    logger.info(
        f"nautilus Sampler created. Chain file: {chain_file if not debug else '(none, debug mode)'}"
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
    logger.info("\n" + "=" * 60)
    logger.info("MCMC COMPLETE (nautilus)")
    logger.info("=" * 60)

    try:
        # equal_weight=True returns an unweighted draw from the posterior
        points, log_w, log_l = sampler_obj.posterior(equal_weight=True)
        logger.info(f"Posterior samples: {points.shape[0]}")
        logger.info(f"log evidence (ln Z): {sampler_obj.log_z:.3f}")

        thinned_filename = chain_file.with_suffix("-samples_thinned.npz")
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
    logger.info("Shutting down SALT2mu subprocesses...")
    list(pool.map(cleanup_worker, range(n_proc)))
    logger.info("All SALT2mu subprocesses terminated.")
    return
