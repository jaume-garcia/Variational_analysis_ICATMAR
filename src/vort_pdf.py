"""
vorticity_pdf.py
================

Computes the empirical PDF (probability density function) of vorticity
for 4 methods/products:

    1. Copernicus (model, "vorticity_model")
    2. LS         ("vorticity_LS", L3 totals obtained by Least Squares)
    3. DIV        ("vorticity_divand_radials", reconstruction from radials via divand)
    4. LS+DIV     combined product ("vorticity_divand_totals"): LS is used where
                  there is valid data and the gaps are filled with the DIV
                  reconstruction (totals)

Vorticity is normalized by the local Coriolis parameter at each grid point,
f = 2*Omega*sin(latitude), with Omega = 7.2921e-5 rad/s, so that all the
PDFs are of zeta/f (dimensionless, Rossby-number-like).

For each of the 4 datasets (using ALL time steps and ALL available grid
points):

    - The normalized histogram (empirical probability density) is computed,
      using all the data from all days.
    - The Gaussian N(mu, sigma) with the sample mean and standard deviation
      of the data themselves is overlaid.
    - The mean, standard deviation, skewness and kurtosis (excess kurtosis,
      Fisher definition: Gaussian -> 0) are computed so that the 4 methods
      can be compared with each other.

Bin width
---------
A bin that is too wide oversmooths the histogram (details are lost, e.g.
the tails); one that is too narrow makes it too noisy. The rule given in
Wilks, D.S. "Statistical Methods in the Atmospheric Sciences"
(2nd ed., section 3.3.5, Eq. 3.12) is used:

        h = c * IQR / n**(1/3)

where IQR is the interquartile range of the data, n the number of data, and
c a constant between 2.0 and 2.6 (Scott, 1992); c=2.6 is optimal for
approximately Gaussian data, and smaller values (towards 2.0) are more
appropriate for skewed or multimodal data -- which is precisely what is
expected for vorticity, where tails associated with extreme events
(intense eddies) are of interest. Hence c=2.0 is used here, a conservative
intermediate point that does not oversmooth the tails. This rule is the
robust version (IQR-based, resistant to outliers) of the Freedman-Diaconis
rule, and is the one recommended in that book for the practical choice of
histogram bin width.

Usage
-----
    python vorticity_pdf.py --input vorticity_methods.nc --outdir figs/

Requires: numpy, scipy, xarray (or netCDF4), matplotlib
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# Angular rotation rate of the Earth [rad/s]
OMEGA_EARTH = 7.2921e-5


# ----------------------------------------------------------------------
# Configuration of the 4 methods to compare
# ----------------------------------------------------------------------
METHOD_LABELS = {
    "cop": "Copernicus",
    "ls": "LS",
    "div": "DIV",
    "ls_div": "LS + DIV",
}

METHOD_COLORS = {
    "cop": "black", #"#2ca02c",
    "ls": "orange", #"#C29B38",
    "div": "red", #"#D62728",
    "ls_div": "blue", #"#1F77B4",
}


def read_vorticity_data(nc_path):
    """
    Reads the NetCDF file with the vorticities of the 4 methods and returns
    a dictionary {key: 1D array without NaN} with ALL time steps and grid
    points stacked.
    """
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        # Local Coriolis parameter f = 2*Omega*sin(lat) at each grid point
        # (dims of "latitude"); xarray broadcasts it to (y, x, time) by
        # dimension name, without depending on the axis order.
        f_local = 2.0 * OMEGA_EARTH * np.sin(np.deg2rad(ds["latitude"]))

        # Normalized vorticity zeta/f (dimensionless)
        vort_model = (ds["vorticity_model"] / f_local).values
        vort_rad = (ds["vorticity_divand_radials"] / f_local).values
        vort_tot = (ds["vorticity_divand_totals"] / f_local).values
        vort_ls = (ds["vorticity_LS"] / f_local).values

    datasets_2d = {
        "cop": vort_model,
        "ls": vort_ls,
        "div": vort_rad,
        "ls_div": vort_tot,
    }

    # Flatten all time steps + grid points into a single 1D vector,
    # removing NaNs (land mask / points without data)
    datasets_1d = {
        key: arr[~np.isnan(arr)].ravel()
        for key, arr in datasets_2d.items()
    }

    return datasets_1d, vort_rad  # radials are also returned in case they
    # need to be inspected separately (not part of the 4 requested PDFs)


def scott_bin_width(data, c=2.0):
    """
    Robust IQR-based bin width, following Wilks (2006),
    "Statistical Methods in the Atmospheric Sciences", Eq. 3.12:

        h = c * IQR / n**(1/3)

    c in [2.0, 2.6]; 2.6 is optimal for Gaussian data (Scott, 1992),
    smaller values are better for skewed / long-tailed data,
    as expected for vorticity.
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = data.size
    h = c * iqr / (n ** (1.0 / 3.0))
    return h


def compute_stats(data, bin_width):
    """Mean, standard deviation, skewness, kurtosis (excess, Fisher) and
    number of bins that will be used in the histogram of this dataset."""
    vmin, vmax = data.min(), data.max()
    nbins = max(10, int(np.ceil((vmax - vmin) / bin_width)))
    return {
        "n_data": data.size,
        "nbins": nbins,
        "mean": np.mean(data),
        "std": np.std(data, ddof=1),
        "skewness": stats.skew(data),
        "kurtosis": stats.kurtosis(data, fisher=True),  # Gaussian -> 0
    }


def common_limits(datasets_1d, bin_width):
    """
    Axis limits common to all plots, so that they are directly comparable:
    X axis = global range of the 4 datasets, Y axis (log) = from the smallest
    non-zero density to the largest of the histograms.
    """
    xmin = min(d.min() for d in datasets_1d.values())
    xmax = max(d.max() for d in datasets_1d.values())
    dmin, dmax = np.inf, 0.0
    for d in datasets_1d.values():
        nb = max(10, int(np.ceil((d.max() - d.min()) / bin_width)))
        counts, _ = np.histogram(d, bins=np.linspace(d.min(), d.max(), nb + 1),
                                 density=True)
        pos = counts[counts > 0]
        dmin, dmax = min(dmin, pos.min()), max(dmax, pos.max())
    return (xmin, xmax), (dmin / 2.0, dmax * 2.0)


def plot_pdf(ax, data, label, color, bin_width, xlim=None, ylim=None):
    """Normalized histogram (empirical PDF) + fitted Gaussian."""
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)

    vmin, vmax = data.min(), data.max()
    nbins = max(10, int(np.ceil((vmax - vmin) / bin_width)))
    bins = np.linspace(vmin, vmax, nbins + 1)

    # 1. Quitar lo de "(data)" en la etiqueta del histograma
    ax.hist(data, bins=bins, density=True, color=color, alpha=0.55,
             edgecolor="none", label=label)

    x = np.linspace(vmin, vmax, 500)
    gauss = stats.norm.pdf(x, loc=mu, scale=sigma)
    
    # 2. Añadir "Gaussian" en la etiqueta de la curva gaussiana
    ax.plot(x, gauss, color="black", linewidth=1.8, linestyle="--",
            #label=f"Gaussian N({mu:.2e}, {sigma:.2e})")
            label=f"Gaussian")

    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"Normalized vorticity $\zeta/f$")
    ax.set_ylabel("PDF")
    #ax.set_title(label)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Vorticity PDF (Copernicus, LS, DIV, LS+DIV)")
    parser.add_argument("--input", "-i",
                         default="../data/january_2026/stats_data/vorticity_methods.nc",
                         help="NetCDF file with the vorticities "
                              "(output of loop_stats_10_days.jl)")
    parser.add_argument("--outdir", "-o",
                         default="../figures/january_2026/physics/vorticity",
                         help="Folder where figures and the statistics table are saved")
    parser.add_argument("--bin-const", type=float, default=2.0,
                         help="Constant c of the bin width rule "
                              "h = c*IQR/n^(1/3) (Wilks Eq. 3.12); "
                              "recommended range 2.0-2.6")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    datasets_1d, _ = read_vorticity_data(args.input)

    # -------------------------------------------------------------
    # Common bin width (so that the 4 PDFs are directly comparable,
    # the SAME bin width is used in all 4), computed with the dataset
    # that has the largest number of points
    # -------------------------------------------------------------
    n_max_key = max(datasets_1d, key=lambda k: datasets_1d[k].size)
    bin_width = scott_bin_width(datasets_1d[n_max_key], c=args.bin_const)

    # Common axis limits for all figures
    xlim, ylim = common_limits(datasets_1d, bin_width)

    stats_table = {}
    for key, data in datasets_1d.items():
        stats_table[key] = compute_stats(data, bin_width)

    print("=" * 78)
    print(f"Common bin width (Wilks/Scott rule, c={args.bin_const}): "
          f"h = {bin_width:.4e} (dimensionless, zeta/f)")
    print("=" * 78)
    header = (f"{'Method':<22}{'n_data':>10}{'n_bins':>9}{'mean':>14}"
              f"{'std':>14}{'skewness':>12}{'kurtosis':>12}")
    print(header)
    print("-" * len(header))
    for key, s in stats_table.items():
        print(f"{METHOD_LABELS[key]:<22}{s['n_data']:>10}{s['nbins']:>9}"
              f"{s['mean']:>14.3e}{s['std']:>14.3e}{s['skewness']:>12.3f}"
              f"{s['kurtosis']:>12.3f}")
    print("=" * 78)

    # Save statistics table to CSV
    stats_csv = os.path.join(args.outdir, "vorticity_stats.csv")
    with open(stats_csv, "w") as f:
        f.write("method,label,n_data,nbins,mean,std,skewness,kurtosis\n")
        for key, s in stats_table.items():
            f.write(f"{key},{METHOD_LABELS[key]},{s['n_data']},{s['nbins']},"
                    f"{s['mean']:.6e},{s['std']:.6e},{s['skewness']:.6f},"
                    f"{s['kurtosis']:.6f}\n")
    print(f"Statistics table saved to: {stats_csv}")

    # -------------------------------------------------------------
    # 4 individual PDFs
    # -------------------------------------------------------------
    for key, data in datasets_1d.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_pdf(ax, data, METHOD_LABELS[key], METHOD_COLORS[key], bin_width,
                 xlim, ylim)
        fig.tight_layout()
        out_path = os.path.join(args.outdir, f"pdf_vorticity_{key}.jpeg")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Figure saved: {out_path}")

    # -------------------------------------------------------------
    # Joint 2x2 figure to compare the 4 methods at a glance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, key in zip(axes.ravel(), ["cop", "ls", "div", "ls_div"]):
        plot_pdf(ax, datasets_1d[key], METHOD_LABELS[key],
                 METHOD_COLORS[key], bin_width, xlim, ylim)
    #fig.suptitle("Vorticity PDF by method (same bin width in all 4)",fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(args.outdir, "pdf_vorticity_overview.jpeg")
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"Comparison figure saved: {combined_path}")

    # -------------------------------------------------------------
    # Figure with the 4 curves overlaid (densities only, no bars)
    # to directly compare shape and tails between methods
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ["cop", "ls", "div", "ls_div"]:
        data = datasets_1d[key]
        vmin, vmax = data.min(), data.max()
        nbins = max(10, int(np.ceil((vmax - vmin) / bin_width)))
        bins = np.linspace(vmin, vmax, nbins + 1)
        counts, edges = np.histogram(data, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, color=METHOD_COLORS[key],
                label=METHOD_LABELS[key], linewidth=1.5)
    #ax.set_xscale("symlog", linthresh=bin_width)  # symmetric: vorticity has a sign
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized vorticity $\zeta/f$ [-]")
    ax.set_ylabel("Probability density")
    ax.set_title("Comparison of empirical vorticity PDFs")
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    overlay_path = os.path.join(args.outdir, "pdf_vorticity_overlay.jpeg")
    fig.savefig(overlay_path, dpi=300)
    plt.close(fig)
    print(f"Overlay figure saved: {overlay_path}")


if __name__ == "__main__":
    main()