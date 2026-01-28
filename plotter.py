"""
DUST2DUST Plotting Module

Contains all plotting and visualization functions for DUST2DUST MCMC analysis.
Separated from main MCMC code for cleaner organization.

Functions:
    - pltting_func: Create chain trace and corner plots from MCMC samples
    - criteria_plotter: Create diagnostic comparison plots (data vs simulation)

Usage as standalone script:
    python plotter.py --chains <chains_file.npz> --outdir <output_dir> [--labels label1,label2,...]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import corner
import argparse
import os

# Default number of color bins (matches DUST2DUST.py)
ncbins = 6


def pltting_func(samples, labels, ndim, outdir, data_input, parameter_overrides=None):
    """
    Create chain trace plots and corner plot from MCMC samples.

    Generates two diagnostic plots:
    1. Trace plot showing parameter evolution over MCMC steps
    2. Corner plot showing parameter correlations and posteriors

    Args:
        samples: MCMC samples array of shape (nsteps, nwalkers, ndim)
        labels: List of parameter labels for plotting
        ndim: Number of dimensions (parameters)
        outdir: Output directory path (with trailing slash)
        data_input: Data input filename (used for output naming)
        parameter_overrides: Dictionary of fixed parameters to exclude from labels (optional)

    Side effects:
        - Saves chain trace plot to outdir/figures/*-chains.pdf
        - Saves corner plot to outdir/figures/*-corner.pdf
        - Prints upload messages for both plots
    """
    # Remove overridden parameters from labels if provided
    if parameter_overrides:
        labels = [l for l in labels if l not in parameter_overrides.keys()]

    # Extract base filename for output
    base_name = data_input.split('.')[0].split('/')[-1]

    # Chain trace plot
    plt.clf()
    fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim), sharex=True)

    # Handle case where ndim=1 (axes won't be an array)
    if ndim == 1:
        axes = [axes]

    for it in range(ndim):
        ax = axes[it]
        ax.plot(samples[:, :, it], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(it)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    chains_path = f'{outdir}figures/{base_name}-chains.pdf'
    plt.savefig(chains_path, bbox_inches='tight')
    print(f'upload {chains_path}')
    plt.close()

    # Corner plot
    flat_samples = samples.reshape(-1, samples.shape[-1])

    plt.clf()
    fig = corner.corner(flat_samples, labels=labels, smooth=True)

    corner_path = f'{outdir}figures/{base_name}-corner.pdf'
    plt.savefig(corner_path)
    print(f'upload {corner_path}')
    plt.close()


def criteria_plotter(chisq, datacount_dict, simcount_dict, poisson_dict, outdir, data_input):
    """
    Create diagnostic plots comparing data and simulation for a given parameter set.

    Creates a 3-panel plot showing:
    - Panel (a): Color histogram comparison
    - Panel (b): Hubble residuals vs color (high/low mass)
    - Panel (c): Hubble residual scatter vs color (high/low mass)

    Each panel includes chi-squared values.

    Args:
        chisq: Dictionary with chi-squared values by component name
               Keys: 'color_hist', 'mures_high', 'mures_low', 'rms_high', 'rms_low'
        datacount_dict: Dictionary with data counts by component name
        simcount_dict: Dictionary with simulation counts by component name
        poisson_dict: Dictionary with Poisson errors by component name
        outdir: Output directory path (with trailing slash)
        data_input: Data input filename (used for output naming)

    Returns:
        str: 'Done' if successful
    """
    cbins = np.linspace(-0.2, 0.25, ncbins)

    # Rescale chisq from log_likelihood (convert from log-likelihood to chi-squared)
    chisq_scaled = {k: -2 * v for k, v in chisq.items()}

    plt.rcParams.update({"text.usetex": True, "font.size": 12})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Panel (a): Color Histogram
    ax = axs[0]
    ax.errorbar(cbins, datacount_dict['color_hist'], yerr=poisson_dict['color_hist'],
                fmt='o', c='darkmagenta', label='Data')
    ax.plot(cbins, simcount_dict['color_hist'], c='dimgray', label='Simulation')
    ax.legend()
    ax.set_xlabel(r'$c$')
    ax.set_ylabel('Count')
    thestring = r'$\chi^2_c =$ ' + str(np.around(chisq_scaled['color_hist'], 1))
    ax.text(-0.2, 50, thestring)
    ax.text(-0.2, 450, 'a)')

    # Panel (b): MURES high and low mass
    ax = axs[1]
    ax.errorbar(cbins, datacount_dict['mures_high'], yerr=poisson_dict['mures_high'],
                fmt='^', c='k', label='Data, High')
    ax.plot(cbins, simcount_dict['mures_high'], c='tab:orange', label='Simulation, High', ls='--')
    ax.errorbar(cbins, datacount_dict['mures_low'], yerr=poisson_dict['mures_low'],
                fmt='s', c='tab:green', label='Data, Low')
    ax.plot(cbins, simcount_dict['mures_low'], c='tab:blue', label='Simulation, Low')
    ax.legend(bbox_to_anchor=[1.7, 1.2], ncol=2)
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$\mu - \mu_{\rm model}$')
    thestring = r'High $\chi^2_{\mu_{\rm res}} =$ ' + str(np.around(chisq_scaled['mures_high'], 1))
    ax.text(-0.2, 0.205, thestring)
    thestring = r'Low $\chi^2_{\mu_{\rm res}} =$ ' + str(np.around(chisq_scaled['mures_low'], 1))
    ax.text(-0.2, 0.18, thestring)
    ax.text(-0.2, 0.275, 'b)')

    # Panel (c): RMS high and low mass
    ax = axs[2]
    ax.errorbar(cbins, datacount_dict['rms_high'], yerr=poisson_dict['rms_high'],
                fmt='^', c='k', label='REAL DATA HIGH')
    ax.plot(cbins, simcount_dict['rms_high'], c='tab:orange', label='SIMULATION HI', ls='--')
    ax.errorbar(cbins, datacount_dict['rms_low'], yerr=poisson_dict['rms_low'],
                fmt='s', c='tab:green', label='REAL DATA LOW')
    ax.plot(cbins, simcount_dict['rms_low'], c='tab:blue', label='SIMULATION LOW')
    ax.set_xlabel(r'$c$')
    ax.set_ylabel(r'$\sigma_{\rm r}$')
    thestring = r'High $\chi^2_{\sigma_{\rm r}} =$ ' + str(np.around(chisq_scaled['rms_high'], 1))
    ax.text(-0.2, 0.42, thestring)
    thestring = r'Low $\chi^2_{\sigma_{\rm r}} =$ ' + str(np.around(chisq_scaled['rms_low'], 1))
    ax.text(-0.2, 0.395, thestring)
    ax.text(-0.2, 0.48, 'c)')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)

    # Save figure
    base_name = data_input.split('.')[0].split('/')[-1]
    fig_path = f'{outdir}figures/{base_name}overplot_observed_DATA_SIM_OVERVIEW.pdf'
    plt.savefig(fig_path, pad_inches=0.01, bbox_inches='tight')
    print(f'upload {fig_path}')
    plt.close()

    return 'Done'


def get_args():
    """Parse command-line arguments for standalone plotting."""
    parser = argparse.ArgumentParser(
        description='DUST2DUST Plotting: Create diagnostic plots from MCMC chains'
    )
    parser.add_argument("--chains", type=str, required=True,
                        help="Path to chains file (.npz)")
    parser.add_argument("--outdir", type=str, default='.',
                        help="Output directory for plots (default: current directory)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated parameter labels (optional)")
    parser.add_argument("--name", type=str, default='mcmc',
                        help="Base name for output files (default: mcmc)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Ensure output directory exists
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not args.outdir.endswith('/'):
        args.outdir += '/'

    # Ensure figures subdirectory exists
    fig_dir = os.path.join(args.outdir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Load chains
    print(f'Loading chains from: {args.chains}')
    chains_data = np.load(args.chains)
    samples = chains_data.f.arr_0

    # Get dimensions
    ndim = samples.shape[-1]
    print(f'  Shape: {samples.shape}')
    print(f'  Dimensions: {ndim}')

    # Parse labels if provided
    if args.labels:
        labels = args.labels.split(',')
        if len(labels) != ndim:
            print(f'WARNING: {len(labels)} labels provided but {ndim} dimensions found.')
            labels = [f'param_{i}' for i in range(ndim)]
    else:
        labels = [f'param_{i}' for i in range(ndim)]

    # Create plots
    print('Creating chain trace and corner plots...')
    pltting_func(samples, labels, ndim, args.outdir, args.name)
    print('Done.')
