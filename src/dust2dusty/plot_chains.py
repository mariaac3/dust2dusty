import argparse
import io
import math
import os
import shutil
import subprocess

import emcee
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Distribution parameter specs (mirrors cli.py Config.DISTRIBUTION_PARAMETERS)
_DISTRIBUTION_PARAMETERS = {
    "Gaussian": ["mu", "std"],
    "Skewed Gaussian": ["mu", "std_l", "std_r"],
    "Exponential": ["Tau"],
    "LogNormal": ["ln_mu", "ln_std"],
    "Double Gaussian": ["a1", "mu1", "std1", "mu2", "std2"],
}


def names_from_config(config_path: str) -> list[str]:
    """Extract expanded parameter names from a dust2dusty YAML config file."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    fitted_params = cfg["FITTED_PARAMS"]
    param_dists = cfg["PARAM_DISTS"]
    splitdict = cfg.get("SPLITDICT", {})

    from dust2dusty.utils import pconv

    return pconv(fitted_params, param_dists, splitdict, _DISTRIBUTION_PARAMETERS)


def fmt_val(v, eu, el):
    """Auto-format value and asymmetric errors to 2 significant figures."""
    scale = min(abs(eu), abs(el))
    if scale == 0:
        return f"{v:.4g}", f"+{eu:.2g}", f"-{el:.2g}"
    mag = math.floor(math.log10(scale))
    dec = max(0, -mag + 1)
    return f"{v:.{dec}f}", f"+{eu:.{dec}f}", f"-{el:.{dec}f}"


def print_table(param_names, flat_chain):
    """Print a summary table of median ± 1-sigma estimates to the terminal."""
    q16, q50, q84 = np.percentile(flat_chain, [15.87, 50, 84.13], axis=0)
    eu = q84 - q50
    el = q50 - q16

    rows = []
    for i, name in enumerate(param_names):
        v, s_up, s_lo = fmt_val(q50[i], eu[i], el[i])
        rows.append((name, v, s_up, s_lo))

    # Column widths
    w0 = max(len("Parameter"), max(len(r[0]) for r in rows))
    w1 = max(len("Median"), max(len(r[1]) for r in rows))
    w2 = max(len("+1σ"), max(len(r[2]) for r in rows))
    w3 = max(len("-1σ"), max(len(r[3]) for r in rows))

    sep = f"+-{'-' * w0}-+-{'-' * w1}-+-{'-' * w2}-+-{'-' * w3}-+"
    hdr = f"| {'Parameter':<{w0}} | {'Median':>{w1}} | {'+1σ':>{w2}} | {'-1σ':>{w3}} |"

    print("\nParameter estimates (median ± 1σ):")
    print(sep)
    print(hdr)
    print(sep)
    for name, v, s_up, s_lo in rows:
        print(f"| {name:<{w0}} | {v:>{w1}} | {s_up:>{w2}} | {s_lo:>{w3}} |")
    print(sep)

    return q50, eu, el


def plot_chains(backend_path, output=None, param_names=None, discard=0, thin=1, ncols=3):
    if not os.path.exists(backend_path):
        raise FileNotFoundError(f"Backend file not found: {backend_path}")

    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    n_steps, n_walkers, n_params = backend.get_chain().shape
    print(f"Steps: {n_steps}  |  Walkers: {n_walkers}  |  Params: {n_params}")

    chain = backend.get_chain(discard=discard, thin=thin)
    log_prob = backend.get_log_prob(discard=discard, thin=thin)
    flat_chain = backend.get_chain(discard=discard, thin=thin, flat=True)

    n_shown = chain.shape[0]
    steps = np.arange(discard, discard + n_shown * thin, thin)

    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]
    elif len(param_names) != n_params:
        raise ValueError(f"Expected {n_params} param names, got {len(param_names)}")

    alpha = max(0.04, min(0.45, 20 / n_walkers))

    ncols_grid = min(ncols, n_params)
    nrows_grid = math.ceil(n_params / ncols_grid)
    fig_w = max(11, ncols_grid * 3.5)
    fig_h = nrows_grid * 2.5 + 2.5 + min(2.0 + n_params * 0.22, 5.0)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── Outer layout: chains top, logprob middle, table bottom ───────────────
    outer = gridspec.GridSpec(
        2,
        1,
        height_ratios=[nrows_grid * 2.5, 2.0],
        hspace=0.45,
        figure=fig,
    )
    param_gs = gridspec.GridSpecFromSubplotSpec(
        nrows_grid,
        ncols_grid,
        subplot_spec=outer[0],
        hspace=0.55,
        wspace=0.35,
    )

    # ── Parameter chain panels ────────────────────────────────────────────────
    for i in range(n_params):
        r, c = divmod(i, ncols_grid)
        ax = fig.add_subplot(param_gs[r, c])
        ax.plot(steps, chain[:, :, i], color="steelblue", alpha=alpha, lw=0.5, rasterized=True)
        ax.set_title(param_names[i], fontsize=9, pad=3)
        ax.set_xlabel("Step", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(6)
        if discard > 0:
            ax.axvline(discard, color="tomato", ls="--", lw=0.8)

    for j in range(n_params, nrows_grid * ncols_grid):
        r, c = divmod(j, ncols_grid)
        fig.add_subplot(param_gs[r, c]).set_visible(False)

    # ── Log-prob strip ────────────────────────────────────────────────────────
    ax_lp = fig.add_subplot(outer[1])
    ax_lp.plot(steps, log_prob, color="darkorange", alpha=alpha, lw=0.5, rasterized=True)
    ax_lp.set_title("log probability", fontsize=9, pad=3)
    ax_lp.set_xlabel("Step", fontsize=8, labelpad=2)
    ax_lp.tick_params(labelsize=7)
    if discard > 0:
        ax_lp.axvline(discard, color="tomato", ls="--", lw=0.8, label=f"burn-in ({discard} steps)")
        ax_lp.legend(fontsize=7, loc="lower right", framealpha=0.6)

    # ── Estimates table (terminal) ────────────────────────────────────────────
    print_table(param_names, flat_chain)

    # ── Suptitle ──────────────────────────────────────────────────────────────
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
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot emcee walker chains + estimates.")
    parser.add_argument("backend")
    parser.add_argument(
        "-o", "--output", default=None, help="Save to PNG file; omit to display inline with imgcat"
    )
    parser.add_argument("-n", "--names", nargs="*", default=None)
    parser.add_argument(
        "--config", default=None, help="Path to dust2dusty YAML config to auto-derive param names"
    )
    parser.add_argument("-d", "--discard", type=int, default=0)
    parser.add_argument("-t", "--thin", type=int, default=1)
    parser.add_argument("-c", "--ncols", type=int, default=3)
    args = parser.parse_args()

    param_names = args.names
    if param_names is None and args.config is not None:
        param_names = names_from_config(args.config)

    plot_chains(
        backend_path=args.backend,
        output=args.output,
        param_names=param_names,
        discard=args.discard,
        thin=args.thin,
        ncols=args.ncols,
    )


if __name__ == "__main__":
    main()
