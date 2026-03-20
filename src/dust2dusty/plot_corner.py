"""
Corner plot (posterior contours) for DUST2DUSTY MCMC chains.

Reads an emcee HDF5 backend file and produces a corner plot using the
``corner`` package.  Parameter names are read from the HDF5 file's
``parameter_names`` attribute and log-sampled parameters (containing
'tau' or 'std') are exp-transformed before plotting.

CLI usage:
    plot_corner <backend.h5> [-o output.png] [-n p1 p2 ...] [-d discard] [-t thin]
                             [--quantiles] [--no-quantiles]
"""

import argparse
import io
import os
import shutil
import subprocess

import corner
import emcee
import numpy as np

from dust2dusty.plot_chains import apply_exp, names_from_hdf5


def plot_corner(
    backend_path,
    output=None,
    param_names=None,
    discard=0,
    thin=1,
    quantiles=(0.1587, 0.5, 0.8413),
):
    """
    Produce a corner plot from an emcee HDF5 backend.

    Args:
        backend_path: Path to the emcee HDF5 backend file.
        output: If given, save the figure to this PNG path; otherwise display
            inline using imgcat (raises RuntimeError if imgcat is not found).
        param_names: List of parameter name strings.  If None, names are read
            from the HDF5 ``parameter_names`` attribute; falls back to
            ``['p0', 'p1', ...]`` if absent.
        discard: Number of initial steps to discard (burn-in).
        thin: Thinning factor applied when reading the chain.
        quantiles: Quantiles shown on the 1-D marginal histograms (default: 1σ).

    Raises:
        FileNotFoundError: If backend_path does not exist.
        ValueError: If the length of param_names does not match the chain dimension.
        RuntimeError: If no output path is given and imgcat is not on PATH.
    """
    if not os.path.exists(backend_path):
        raise FileNotFoundError(f"Backend file not found: {backend_path}")

    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    n_steps, n_walkers, n_params = backend.get_chain().shape
    print(f"Steps: {n_steps}  |  Walkers: {n_walkers}  |  Params: {n_params}")

    chain = backend.get_chain(discard=discard, thin=thin)
    flat_chain = backend.get_chain(discard=discard, thin=thin, flat=True)

    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]
    elif len(param_names) != n_params:
        raise ValueError(f"Expected {n_params} param names, got {len(param_names)}")

    chain, flat_chain = apply_exp(chain, flat_chain, param_names)

    # 1σ and 2σ enclosed fractions for a 2D Gaussian: 1 - exp(-n²/2)
    levels_2d = 1 - np.exp(-0.5 * np.array([1, 2]) ** 2)  # ≈ [0.393, 0.865]

    fig = corner.corner(
        flat_chain,
        labels=param_names,
        levels=levels_2d,
        quantiles=list(quantiles),
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 9},
        title_fmt=".3g",
    )
    fig.suptitle(
        f"{os.path.basename(backend_path)}   —   "
        f"{n_walkers} walkers · {n_steps} steps · {n_params} params",
        fontsize=10,
        y=1.01,
    )

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {output}")
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        if shutil.which("imgcat"):
            subprocess.run(["imgcat"], input=buf.read(), check=True)
        else:
            raise RuntimeError(
                "imgcat not found in PATH. Install it from "
                "https://iterm2.com/documentation-images.html "
                "or pass -o <file> to save instead."
            )

    import matplotlib.pyplot as plt

    plt.close(fig)


def main():
    """Entry point for the plot_corner command."""
    parser = argparse.ArgumentParser(description="Corner plot of posterior from emcee HDF5 backend.")
    parser.add_argument("backend")
    parser.add_argument(
        "-o", "--output", default=None, help="Save to PNG file; omit to display inline with imgcat"
    )
    parser.add_argument(
        "-n", "--names", nargs="*", default=None, help="Override param names (read from HDF5 by default)"
    )
    parser.add_argument("-d", "--discard", type=int, default=0)
    parser.add_argument("-t", "--thin", type=int, default=1)
    parser.add_argument(
        "--no-quantiles",
        dest="quantiles",
        action="store_false",
        help="Suppress quantile lines on 1-D marginals",
    )
    parser.set_defaults(quantiles=True)
    args = parser.parse_args()

    param_names = args.names
    if param_names is None:
        param_names = names_from_hdf5(args.backend)

    quantiles = (0.1587, 0.5, 0.8413) if args.quantiles else ()

    plot_corner(
        backend_path=args.backend,
        output=args.output,
        param_names=param_names,
        discard=args.discard,
        thin=args.thin,
        quantiles=quantiles,
    )


if __name__ == "__main__":
    main()
