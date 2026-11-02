"""
1D kinetic energy spectra along transects for 4 surface-current reconstructions
(Copernicus reference, DIV, LS+DIV, LS) -- version 2.

Changes with respect to spectrum_energy.py
------------------------------------------
1. COMMON MASK / COMMON TIME STEPS: a snapshot is used only if ALL methods have
   100 % finite u,v along the transect. Nothing is filled with the mean, so no
   artificial steps are injected into the spectra. The same snapshots are used
   for the four methods (and for the error spectra).
2. ERROR SPECTRA: spectrum of (u_method - u_ref, v_method - v_ref), processed
   exactly like the velocity spectra. Reported as absolute spectrum and
   relative to the reference energy, E_err(k) / E_ref(k).
3. UNITS: density per rad/km (extra 1/2pi), consistent with the x axis.
4. RESOLVED SCALE: reports both the last wavelength still above the threshold
   and the first wavelength below it (the effective resolution lies between).
5. Barycentric weights computed once per transect (much faster than building a
   LinearNDInterpolator for every time step).
6. Summary table printed and saved as CSV (values for the paper).

Usage:  python spectrum_energy_v2.py [velocity_methods.nc]
"""

import os
import sys
import csv
import numpy as np
import xarray as xr
from scipy.signal.windows import hann
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


CONFIG = {
    "velocity_file": "../data/january_2026/stats_data/velocity_methods.nc",
    "output_dir": "../figures/january_2026/physics",
    "output_name": "ke_spectrum_1D_transect.jpeg",

    "transects": [
        {"name": "Transect 1 - SE (anticyclonic eddy)",
         "lon1": 2., "lat1": 40.75, "lon2": 3.5, "lat2": 41.75},
        {"name": "Transect 2 - center, coast-parallel (NE-SW)",
         "lon1": 2.0, "lat1": 41.10, "lon2": 3.0, "lat2": 41.60},
        {"name": "Transect 3 - zonal, towards the coast (E-W)",
         "lon1": 1.9, "lat1": 40.90, "lon2": 2.7, "lat2": 40.90},
    ],

    "transect_dx_km": None,        # None -> native grid spacing
    "per_rad": True,               # density per rad/km (True) or per cycle/km (False)

    "variables": {
        "Copernicus": ("u_model",          "v_model"),
        "DIV":        ("u_divand_radials", "v_divand_radials"),
        "LS+DIV":     ("u_divand_totals",  "v_divand_totals"),
        "LS":         ("u_LS",             "v_LS"),
    },
    "colors": {"Copernicus": "black", "DIV": "red", "LS+DIV": "blue", "LS": "orange"},

    "reference": "Copernicus",
    "resolved_scale_threshold": 0.5,
}


# =====================================================================
# GEOMETRY
# =====================================================================

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def native_grid_dx_km(lon2d, lat2d):
    lat0 = np.nanmean(lat2d)
    km_lon = 111.32 * np.cos(np.deg2rad(lat0))
    km_lat = 111.32
    out = []
    for ax in (0, 1):
        dlon = np.nanmedian(np.abs(np.diff(lon2d, axis=ax)))
        dlat = np.nanmedian(np.abs(np.diff(lat2d, axis=ax)))
        out.append(np.hypot(dlon * km_lon, dlat * km_lat))
    return float(np.nanmin(out))


def build_transect(lon1, lat1, lon2, lat2, cfg, lon2d, lat2d):
    total_km = haversine_km(lon1, lat1, lon2, lat2)
    dx_km = cfg["transect_dx_km"] or native_grid_dx_km(lon2d, lat2d)
    n = max(int(round(total_km / dx_km)), 16)
    t = np.linspace(0.0, 1.0, n)
    lon_t = lon1 + t * (lon2 - lon1)
    lat_t = lat1 + t * (lat2 - lat1)
    ds_km = (t[1] - t[0]) * total_km
    print(f"  Transect: {total_km:.1f} km, {n} points, ds={ds_km:.3f} km "
          f"(native grid spacing ~{dx_km:.3f} km)")
    return lon_t, lat_t, ds_km, total_km


def transect_weights(tri, lon_t, lat_t):
    """Barycentric interpolation weights of the transect points inside the
    Delaunay triangulation (computed once per transect)."""
    pts = np.column_stack([lon_t, lat_t])
    simplex = tri.find_simplex(pts)
    outside = simplex < 0
    s = np.where(outside, 0, simplex)
    T = tri.transform[s]
    b = np.einsum("pij,pj->pi", T[:, :2, :], pts - T[:, 2, :])
    w = np.column_stack([b, 1.0 - b.sum(axis=1)])
    verts = tri.simplices[s]
    return verts, w, outside


def sample_on_transect(field, verts, w, outside):
    """field: (nt, ny, nx) -> (nt, npts). NaN at any vertex -> NaN."""
    nt = field.shape[0]
    flat = field.reshape(nt, -1)
    out = (flat[:, verts] * w[None, :, :]).sum(axis=-1)
    out[:, outside] = np.nan
    return out


# =====================================================================
# SPECTRA
# =====================================================================

def detrend_and_window_1d(x):
    n = len(x)
    idx = np.arange(n)
    A = np.c_[idx, np.ones(n)]
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    x_dt = x - A @ coef
    win = hann(n)
    return x_dt * win, np.mean(win ** 2)


def compute_1d_ke_spectrum(u, v, ds_km, per_rad=True):
    n = len(u)
    u_win, c = detrend_and_window_1d(u)
    v_win, _ = detrend_and_window_1d(v)
    U = np.fft.rfft(u_win) / n
    V = np.fft.rfft(v_win) / n
    k = 2 * np.pi * np.fft.rfftfreq(n, d=ds_km)
    L = n * ds_km
    KE = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2) * L / c
    if n % 2 == 0:
        KE[1:-1] *= 2
    else:
        KE[1:] *= 2
    if per_rad:
        KE /= (2 * np.pi)
    return k, KE


def mean_spectrum(u_t, v_t, ds_km, per_rad):
    specs = []
    k = None
    for i in range(u_t.shape[0]):
        k, KE = compute_1d_ke_spectrum(u_t[i], v_t[i], ds_km, per_rad)
        specs.append(KE)
    return k, np.mean(specs, axis=0)


def spectra_for_transect(aligned, tri, tsect, cfg, lon2d, lat2d):
    lon_t, lat_t, ds_km, total_km = build_transect(
        tsect["lon1"], tsect["lat1"], tsect["lon2"], tsect["lat2"], cfg, lon2d, lat2d)
    verts, w, outside = transect_weights(tri, lon_t, lat_t)

    samples = {}
    for label, (u_all, v_all) in aligned.items():
        if u_all is None:
            continue
        samples[label] = (sample_on_transect(u_all, verts, w, outside),
                          sample_on_transect(v_all, verts, w, outside))

    # ---- common time steps: 100 % valid points for ALL methods
    nt = next(iter(samples.values()))[0].shape[0]
    valid = np.ones(nt, dtype=bool)
    for label, (u_t, v_t) in samples.items():
        ok = np.isfinite(u_t).all(axis=1) & np.isfinite(v_t).all(axis=1)
        print(f"    {label:11s}: {ok.sum():4d}/{nt} snapshots fully valid on the transect")
        valid &= ok
    n_common = int(valid.sum())
    print(f"  -> COMMON snapshots (all methods, 100% valid): {n_common}/{nt}")
    if n_common < 5:
        print("  -> too few common snapshots -> transect skipped")
        return None, None, 0

    ref = cfg["reference"]
    spec, err = {}, {}
    for label, (u_t, v_t) in samples.items():
        spec[label] = mean_spectrum(u_t[valid], v_t[valid], ds_km, cfg["per_rad"])
        if label != ref:
            du = u_t[valid] - samples[ref][0][valid]
            dv = v_t[valid] - samples[ref][1][valid]
            err[label] = mean_spectrum(du, dv, ds_km, cfg["per_rad"])
    return spec, err, n_common


def interp_loglog(k_common, k, E):
    out = np.full_like(k_common, np.nan, dtype=float)
    if k is None or E is None:
        return out
    ok = (k > 0) & np.isfinite(E) & (E > 0)
    if ok.sum() < 2:
        return out
    inside = (k_common >= k[ok].min()) & (k_common <= k[ok].max())
    out[inside] = np.exp(np.interp(np.log(k_common[inside]), np.log(k[ok]), np.log(E[ok])))
    return out


def average_across_transects(list_of_dicts):
    """list of {label: (k, E)} -> {label: (k_common, E_mean)} (log-log interp
    onto the k grid of the shortest transect)."""
    list_of_dicts = [d for d in list_of_dicts if d]
    k_common = min((k for d in list_of_dicts for k, _ in d.values()), key=len)
    labels = list(list_of_dicts[0].keys())
    out = {}
    for lab in labels:
        stack = [interp_loglog(k_common, *d[lab]) for d in list_of_dicts if lab in d]
        out[lab] = (k_common, np.nanmean(np.array(stack), axis=0))
    return out


# =====================================================================
# RESOLVED SCALE + SUMMARY TABLE
# =====================================================================

def resolved_scale_bracket(k, E_ref, E_m, thr):
    """(last wavelength with E_m/E_ref >= thr before the first drop,
        first wavelength with ratio < thr). Scans from large to small scales."""
    ok = k > 0
    kk, ratio = k[ok], (E_m / E_ref)[ok]
    below = np.where(ratio < thr)[0]
    if len(below) == 0:
        return None, None
    i = below[0]
    wl_below = 2 * np.pi / kk[i]
    wl_ok = 2 * np.pi / kk[i - 1] if i > 0 else None
    return wl_ok, wl_below


def print_and_save_summary(avg_spec, avg_err, cfg, csv_path):
    ref = cfg["reference"]
    k, E_ref = avg_spec[ref]
    methods = [m for m in avg_spec if m != ref]

    head = f"{'lambda[km]':>10s} {'k[rad/km]':>10s} {'E_ref':>10s} | " + \
           " ".join(f"{m+' E/Eref':>14s}" for m in methods) + " | " + \
           " ".join(f"{m+' Eerr/Eref':>16s}" for m in methods)
    print("\n" + head)
    rows = []
    for i in range(len(k)):
        if k[i] <= 0:
            continue
        wl = 2 * np.pi / k[i]
        r = [avg_spec[m][1][i] / E_ref[i] for m in methods]
        e = [avg_err[m][1][i] / E_ref[i] for m in methods]
        print(f"{wl:10.1f} {k[i]:10.3f} {E_ref[i]:10.3e} | " +
              " ".join(f"{x:14.2f}" for x in r) + " | " + " ".join(f"{x:16.2f}" for x in e))
        rows.append([wl, k[i], E_ref[i]] + r + e)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["lambda_km", "k_rad_km", "E_ref"] +
                    [f"{m}_E_over_Eref" for m in methods] +
                    [f"{m}_Eerr_over_Eref" for m in methods])
        wr.writerows(rows)
    print(f"  Table saved: {csv_path}")

    thr = cfg["resolved_scale_threshold"]
    print(f"\nRESOLVED SCALE (E_method >= {thr*100:.0f}% of reference), mean of transects")
    for m in methods:
        wl_ok, wl_b = resolved_scale_bracket(k, E_ref, avg_spec[m][1], thr)
        if wl_b is None:
            print(f"  {m:8s}: never below threshold in the resolved range")
        else:
            ok_txt = f"{wl_ok:.1f}" if wl_ok else "n/a"
            print(f"  {m:8s}: above threshold down to lambda = {ok_txt} km; "
                  f"below at lambda = {wl_b:.1f} km")

    print("\nWavelength where the ERROR energy reaches 50% / 100% of the reference energy")
    for m in methods:
        rel = avg_err[m][1] / E_ref
        for level in (0.5, 1.0):
            idx = np.where((rel >= level) & (k > 0))[0]
            txt = f"{2*np.pi/k[idx[0]]:.1f} km" if len(idx) else "never"
            print(f"  {m:8s}: Eerr/Eref >= {level:.1f} first at lambda = {txt}")


# =====================================================================
# PLOTS
# =====================================================================


def _wl_axis(ax, lp=np.array([70., 40., 30., 20., 15., 10., 7.])):
    f = lambda k: 2 * np.pi / np.where(k == 0, np.nan, k)
    top = ax.secondary_xaxis("top", functions=(f, f))

    # los ticks se dan directamente en km (unidades del eje secundario)
    top.set_xticks(lp)
    top.set_xticklabels([f"{l:g}" for l in lp])
    top.xaxis.set_minor_locator(plt.NullLocator())

    top.set_xlabel("wavelength [km]", fontsize=13)


def ylabel(cfg):
    return (r"$E(k)$ [m$^2$ s$^{-2}$ (rad km$^{-1}$)$^{-1}$]" if cfg["per_rad"]
            else r"$E(k)$ [m$^2$ s$^{-2}$ (cycle km$^{-1}$)$^{-1}$]")


def plot_spectra(results, cfg, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    kref = Eref = None
    for label, (k, E) in results.items():
        v = (k > 0) & np.isfinite(E) & (E > 0)
        ax.loglog(k[v], E[v], "-o", ms=6.5, color=cfg["colors"][label], lw=1.5, label=label)
        if label == cfg["reference"]:
            kref, Eref = k[v], E[v]
    if kref is not None and len(kref) > 5:
        k0 = kref[len(kref) // 3]
        E0 = np.interp(k0, kref, Eref)
        kk = np.array([kref.min(), kref.max()])
        #ax.loglog(kk, E0 * (kk / k0) ** -2, "--", color="cyan", label=r"$k^{-2}$")
        ax.loglog(kk, E0 * (kk / k0) ** -3, "--", color="dimgray", label=r"$k^{-3}$")
        #ax.loglog(kk, E0 * (kk / k0) ** -5.5, "--", color="purple", label=r"$k^{-5.5}$")
        #ax.loglog(kk, E0 * (kk / k0) ** -4, "--", color="green", label=r"$k^{-4}$")
        ax.loglog(kk, E0 * (kk / k0) ** (-5 / 3), ":", color="gray", label=r"$k^{-5/3}$")
    ax.set_xlabel("wavenumber [rad/km]", fontsize=15)
    ax.set_ylabel(ylabel(cfg), fontsize=15)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=15)
    _wl_axis(ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Figure saved: {path}")


def plot_error_spectra(avg_spec, avg_err, cfg, path):
    ref = cfg["reference"]
    k, E_ref = avg_spec[ref]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    a1.loglog(k[k > 0], E_ref[k > 0], "-o", ms=4, color="black", label=f"{ref} (signal)")
    for m, (ke, Ee) in avg_err.items():
        a1.loglog(ke[ke > 0], Ee[ke > 0], "-o", ms=4, color=cfg["colors"][m],
                  label=f"error {m}")
        a2.semilogx(ke[ke > 0], Ee[ke > 0] / E_ref[ke > 0], "-o", ms=4,
                    color=cfg["colors"][m], label=m)
    a2.axhline(1.0, color="gray", ls="--", lw=1)
    a2.axhline(0.5, color="gray", ls=":", lw=1)
    a1.set_xlabel("wavenumber [rad/km]", fontsize=13)
    a2.set_xlabel("wavenumber [rad/km]", fontsize=13)
    a1.set_ylabel(ylabel(cfg), fontsize=13)
    a2.set_ylabel(r"$E_{err}(k)\,/\,E_{ref}(k)$", fontsize=13)
    for a in (a1, a2):
        a.grid(True, which="both", ls=":", alpha=0.4)
        a.legend(fontsize=13)
    _wl_axis(a1)
    _wl_axis(a2)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Figure saved: {path}")


# =====================================================================
# MAIN
# =====================================================================

def align_to_grid_dims(da, lon_dims):
    other = [d for d in da.dims if d not in lon_dims]
    if len(other) != 1:
        raise ValueError(f"Expected one time dim apart from {lon_dims}, got {da.dims}")
    return da.transpose(other[0], *lon_dims)


def main():
    cfg = CONFIG
    if len(sys.argv) > 1:
        cfg["velocity_file"] = sys.argv[1]
    if len(sys.argv) > 2:
        cfg["output_dir"] = sys.argv[2]

    ds = xr.open_dataset(cfg["velocity_file"])
    lon_dims = ds["longitude"].dims
    lon2d, lat2d = ds["longitude"].values, ds["latitude"].values
    print(f"Spatial dims {lon_dims}, shape {lon2d.shape}")
    tri = Delaunay(np.column_stack([lon2d.ravel(), lat2d.ravel()]))

    aligned = {}
    for label, (uv, vv) in cfg["variables"].items():
        if uv not in ds or vv not in ds:
            print(f"  {label}: '{uv}'/'{vv}' missing -> skipped")
            continue
        aligned[label] = (align_to_grid_dims(ds[uv], lon_dims).values,
                          align_to_grid_dims(ds[vv], lon_dims).values)

    base, ext = os.path.splitext(cfg["output_name"])
    all_spec, all_err, n_used = [], [], []
    for it, ts in enumerate(cfg["transects"], 1):
        print(f"\n{'=' * 60}\nTransect {it}/{len(cfg['transects'])}: {ts['name']}\n{'=' * 60}")
        spec, err, n = spectra_for_transect(aligned, tri, ts, cfg, lon2d, lat2d)
        if spec is None:
            continue
        all_spec.append(spec)
        all_err.append(err)
        n_used.append(n)
        plot_spectra(spec, cfg, os.path.join(cfg["output_dir"], f"{base}_t{it}{ext}"))

    print(f"\n{'#' * 60}\nMEAN OF {len(all_spec)} TRANSECTS "
          f"(common snapshots per transect: {n_used})\n{'#' * 60}")
    avg_spec = average_across_transects(all_spec)
    avg_err = average_across_transects(all_err)
    plot_spectra(avg_spec, cfg, os.path.join(cfg["output_dir"], f"{base}_mean{ext}"))
    plot_error_spectra(avg_spec, avg_err, cfg,
                       os.path.join(cfg["output_dir"], f"{base}_error_mean{ext}"))
    print_and_save_summary(avg_spec, avg_err, cfg,
                           os.path.join(cfg["output_dir"], "ke_spectrum_summary.csv"))
    ds.close()


if __name__ == "__main__":
    main()