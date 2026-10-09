import numpy as np
from scipy.interpolate import griddata
import math

# -----------------------------------------------------------------------
# Module: mathematics.py
# Purpose: Mathematical utilities for oceanographic data processing,
#          including spatial interpolation, statistical error metrics,
#          and differential operators (vorticity, divergence, kinetic energy).
# -----------------------------------------------------------------------


def interp_grid_vel(lon_high_res_grid, lat_high_res_grid, u_high_res_grid, v_high_res_grid,
                    lon_coarse_grid, lat_coarse_grid, mask_coarse_grid=[], points_mode=False):
    """
    Interpolates velocity components (u, v) from a high-resolution grid to a
    coarser grid (or to a set of scattered points) using linear interpolation.

    Parameters:
    -----------
    lon_high_res_grid, lat_high_res_grid : ndarray
        Longitude and latitude of the high-resolution source grid
    u_high_res_grid, v_high_res_grid : ndarray
        East and north velocity components on the high-resolution grid
    lon_coarse_grid, lat_coarse_grid : ndarray
        Target grid coordinates (1D or 2D); or 1D point arrays if points_mode=True
    mask_coarse_grid : ndarray, optional
        Land/sea mask applied to the interpolated values (only in grid mode)
    points_mode : bool
        If True, interpolate to irregular scattered points (1D arrays);
        if False, interpolate to a regular or 2D grid

    Returns:
    --------
    u_interp, v_interp : ndarray
        Interpolated east and north velocity components on the target grid/points
    """
    # Flatten source grid to a list of (lon, lat) points
    points = np.column_stack((lon_high_res_grid.flatten(), lat_high_res_grid.flatten()))
    u_values = u_high_res_grid.flatten()
    v_values = v_high_res_grid.flatten()

    # Remove NaN values from the source data before interpolating
    valid_indices = ~np.isnan(u_values) & ~np.isnan(v_values)
    valid_points = points[valid_indices]
    valid_u = u_values[valid_indices]
    valid_v = v_values[valid_indices]

    # Handle masked arrays by filling masked values with NaN
    if hasattr(valid_u, 'filled'):
        valid_u = valid_u.filled(np.nan)
    if hasattr(valid_v, 'filled'):
        valid_v = valid_v.filled(np.nan)

    if points_mode:
        # Interpolate to irregular 1D scattered points
        xi = np.column_stack((lon_coarse_grid, lat_coarse_grid))
        shape_out = lon_coarse_grid.shape

    else:
        # Build 2D meshgrid if 1D coordinate arrays are provided
        if lon_coarse_grid.ndim == 1 and lat_coarse_grid.ndim == 1:
            lon_mesh_coarse_grid, lat_mesh_coarse_grid = np.meshgrid(lon_coarse_grid, lat_coarse_grid)
        else:
            lon_mesh_coarse_grid, lat_mesh_coarse_grid = lon_coarse_grid, lat_coarse_grid

        xi = np.column_stack((lon_mesh_coarse_grid.flatten(), lat_mesh_coarse_grid.flatten()))
        shape_out = lon_mesh_coarse_grid.shape

    # Perform linear interpolation; points outside the convex hull become NaN
    u_interp = griddata(valid_points, valid_u, xi, method='linear', fill_value=np.nan)
    v_interp = griddata(valid_points, valid_v, xi, method='linear', fill_value=np.nan)

    # Reshape to the target grid shape
    u_interp = u_interp.reshape(shape_out)
    v_interp = v_interp.reshape(shape_out)

    # Apply the land/sea mask (sets land points to zero)
    if len(mask_coarse_grid) != 0 and not points_mode:
        u_interp = u_interp * mask_coarse_grid
        v_interp = v_interp * mask_coarse_grid

    return u_interp, v_interp


# --------------------------------------------------------------


def interp_grid_eta(lon_high_res_grid, lat_high_res_grid, eta_high_res_grid,
                    lon_coarse_grid, lat_coarse_grid, mask_coarse_grid=[]):
    """
    Interpolates the sea surface height (SSH / eta) from a high-resolution grid
    to a coarser grid using linear interpolation.

    Parameters:
    -----------
    lon_high_res_grid, lat_high_res_grid : ndarray
        Longitude and latitude of the high-resolution source grid
    eta_high_res_grid : ndarray
        Sea surface height on the high-resolution grid
    lon_coarse_grid, lat_coarse_grid : ndarray
        Target grid coordinates (1D or 2D)
    mask_coarse_grid : ndarray, optional
        Land/sea mask applied to the interpolated SSH values

    Returns:
    --------
    eta_interp : ndarray
        Interpolated SSH on the target grid
    """
    # Build 2D meshgrid if 1D coordinate arrays are provided
    if lon_coarse_grid.ndim == 1 and lat_coarse_grid.ndim == 1:
        lon_mesh_coarse_grid, lat_mesh_coarse_grid = np.meshgrid(lon_coarse_grid, lat_coarse_grid)
    else:
        lon_mesh_coarse_grid, lat_mesh_coarse_grid = lon_coarse_grid, lat_coarse_grid

    # Flatten the source grid to (lon, lat) point pairs
    points = np.column_stack((lon_high_res_grid.flatten(), lat_high_res_grid.flatten()))

    eta_values = eta_high_res_grid.flatten()

    # Remove NaN values from the source data
    valid_indices = ~np.isnan(eta_values)
    valid_points = points[valid_indices]
    valid_eta = eta_values[valid_indices]

    xi = np.column_stack((lon_mesh_coarse_grid.flatten(), lat_mesh_coarse_grid.flatten()))

    eta_interp = griddata(valid_points, valid_eta, xi, method='linear', fill_value=np.nan)
    eta_interp = eta_interp.reshape(lon_mesh_coarse_grid.shape)

    # Apply the land/sea mask
    if len(mask_coarse_grid) != 0:
        eta_interp = eta_interp * mask_coarse_grid

    return eta_interp


# --------------------------------------------------------------

def putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS,
                               u_array, v_array):
    """
    Assigns scattered data points to the nearest node of a Least Squares (LS) grid.
    Points that do not fall exactly on a grid node are flagged with a warning.

    Parameters:
    -----------
    lon_array : array-like
        Longitudes of the data points
    lat_array : array-like
        Latitudes of the data points
    lon_grid_LS : ndarray
        2D longitude grid of the LS mesh
    lat_grid_LS : ndarray
        2D latitude grid of the LS mesh
    u_array : array-like
        East velocity component at each data point
    v_array : array-like
        North velocity component at each data point

    Returns:
    --------
    u_LS_2d, v_LS_2d : ndarray
        East and north velocity components mapped onto the 2D LS grid (NaN elsewhere)
    """
    tolerance = 1e-6

    # Initialise output arrays with NaN
    u_LS_2d = np.full(lon_grid_LS.shape, np.nan)
    v_LS_2d = np.full(lon_grid_LS.shape, np.nan)

    points_found = 0
    points_not_found = 0

    for j in range(len(lon_array)):
        # Find the closest grid node using Euclidean distance in degree space
        distances = np.sqrt((lon_grid_LS - lon_array[j])**2 +
                            (lat_grid_LS - lat_array[j])**2)
        min_distance = np.min(distances)
        idx_closest = np.unravel_index(np.argmin(distances), distances.shape)
        idx_lon, idx_lat = idx_closest

        # Verify that the closest node matches the data point within tolerance
        lon_closest = lon_grid_LS[idx_lon, idx_lat]
        lat_closest = lat_grid_LS[idx_lon, idx_lat]
        is_grid_point = (abs(lon_closest - lon_array[j]) < tolerance and
                         abs(lat_closest - lat_array[j]) < tolerance)

        if is_grid_point:
            points_found += 1
        else:
            points_not_found += 1

            if points_not_found <= 5:  # Show only the first 5 warnings
                print(f"WARNING: Point {j} is not exactly on the grid")
                print(f"  Distance: {min_distance:.6f}")
                print(f"  Target:   ({lon_array[j]:.6f}, {lat_array[j]:.6f})")
                print(f"  Closest:  ({lon_closest:.6f}, {lat_closest:.6f})")

        # Assign velocities to the closest grid node
        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]

    print(f"\nSummary: {points_found} points found, {points_not_found} points not exact")

    return u_LS_2d, v_LS_2d


# --------------------------------------------------------------


def stat_err(u_model, v_model, u_obs, v_obs, N_obs):
    """
    Calculates statistical error metrics between modelled and observed velocity fields:
    Root Mean Square Error (RMSE) and Mean Bias (MB) for u, v, and total velocity.

    Parameters:
    -----------
    u_model, v_model : ndarray
        Modelled east and north velocity components
    u_obs, v_obs : ndarray
        Observed east and north velocity components
    N_obs : int
        Number of valid observations used for normalisation

    Returns:
    --------
    tuple
        u_rms, v_rms      : RMSE for the u and v components
        u_mb, v_mb        : Mean bias for the u and v components
        total_rms         : Total RMSE (combined u and v)
        total_mb          : Total mean bias
    """
    # Temporal mean of observations and model
    u_mean_obs   = np.nanmean(u_obs)
    v_mean_obs   = np.nanmean(v_obs)
    u_model_mean = np.nanmean(u_model)
    v_model_mean = np.nanmean(v_model)

    # RMSE for each component
    u_rms = np.sqrt(np.nansum((u_obs - u_model)**2) / N_obs)
    v_rms = np.sqrt(np.nansum((v_obs - v_model)**2) / N_obs)

    # Mean bias: difference of anomalies (observation anomaly minus model anomaly)
    u_mb = np.nansum((u_obs - u_mean_obs) - (u_model - u_model_mean)) / N_obs
    v_mb = np.nansum((v_obs - v_mean_obs) - (v_model - v_model_mean)) / N_obs

    # Total (vector) RMSE and mean bias
    total_rms = np.sqrt(np.nansum((u_obs - u_model)**2 + (v_obs - v_model)**2) / N_obs)
    total_mb  = np.nansum(
        (u_obs - u_mean_obs) - (u_model - u_model_mean) +
        (v_obs - v_mean_obs) - (v_model - v_model_mean)
    ) / N_obs

    return u_rms, v_rms, u_mb, v_mb, total_rms, total_mb


# --------------------------------------------------------------


def var_dev(data):
    """
    Calculates the sample variance and standard deviation of a 1D dataset.

    Parameters:
    -----------
    data : array-like
        Input data values (NaNs are ignored in the mean calculation)

    Returns:
    --------
    var : float
        Sample variance (divided by n-1)
    std_dev : float
        Sample standard deviation
    """
    n = len(data)
    data_mean = np.nanmean(data)

    # Sum of squared deviations from the mean
    squared_deviations = [(x - data_mean) ** 2 for x in data]
    sum_sq = np.nansum(squared_deviations)

    # Sample variance (Bessel's correction: divide by n-1)
    var = sum_sq / (n - 1)

    # Sample standard deviation
    std_dev = math.sqrt(var)

    return var, std_dev


# --------------------------------------------------------------


def correlation(u_interp, v_interp, u_model, v_model):
    """
    Calculates Pearson correlation coefficients between interpolated and modelled
    velocity components (u, v) and their magnitudes.

    Parameters:
    -----------
    u_interp, v_interp : array-like
        Interpolated east and north velocity components (e.g. from observations)
    u_model, v_model : array-like
        Modelled east and north velocity components

    Returns:
    --------
    u_corr, v_corr, mag_corr : float
        Pearson correlation coefficient for u, v, and velocity magnitude
    """
    # Build masks to exclude NaN values
    valid_u_mask   = ~np.isnan(u_interp) & ~np.isnan(u_model)
    valid_v_mask   = ~np.isnan(v_interp) & ~np.isnan(v_model)
    valid_mag_mask = (~np.isnan(u_interp) & ~np.isnan(v_interp) &
                      ~np.isnan(u_model)  & ~np.isnan(v_model))

    # Pearson correlation for u and v individually
    u_corr = np.corrcoef(u_interp[valid_u_mask], u_model[valid_u_mask])[0, 1]
    v_corr = np.corrcoef(v_interp[valid_v_mask], v_model[valid_v_mask])[0, 1]

    # Pearson correlation for velocity magnitude
    mag_interp = np.sqrt(u_interp**2 + v_interp**2)
    mag_model  = np.sqrt(u_model**2 + v_model**2)

    mag_corr = np.corrcoef(mag_interp[valid_mag_mask], mag_model[valid_mag_mask])[0, 1]

    print(f"U correlation: {u_corr:.3f}")
    print(f"V correlation: {v_corr:.3f}")
    print(f"Total velocity correlation: {mag_corr:.3f}")

    return u_corr, v_corr, mag_corr


# --------------------------------------------------------------


def vorticity(u, v, lon, lat, nx=None, ny=None):
    """
    Calculates the vertical component of relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
    using centred finite differences on a spherical Earth.
    Accepts both 2D arrays and 1D flattened vectors.

    Parameters:
    -----------
    u, v : ndarray
        East and north velocity components (2D or 1D)
    lon, lat : ndarray
        Longitude and latitude grids (2D or 1D, matching u and v)
    nx : int, optional
        Number of grid points in the x-direction (required for 1D input)
    ny : int, optional
        Number of grid points in the y-direction (required for 1D input)

    Returns:
    --------
    vorticity : ndarray
        Vorticity field (s⁻¹); same shape as input (returns 1D if input was 1D)
    """
    # If inputs are 1D, reshape to 2D for finite-difference calculations
    if u.ndim == 1:
        if nx is None or ny is None:
            raise ValueError("For 1D vectors, please provide grid dimensions nx and ny")

        u   = u.reshape(ny, nx)
        v   = v.reshape(ny, nx)
        lon = lon.reshape(ny, nx)
        lat = lat.reshape(ny, nx)

        return_1d = True
    else:
        return_1d = False

    ny, nx = u.shape
    vorticity = np.full_like(u, np.nan)

    R = 6.371e6  # Earth's radius in metres

    print(nx, ny)

    # Central differences on interior grid points (skip boundary rows/columns)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # Skip grid points with any NaN neighbour
            if (np.isnan(u[i, j]) or np.isnan(v[i, j]) or
                    np.isnan(u[i+1, j]) or np.isnan(u[i-1, j]) or
                    np.isnan(v[i, j+1]) or np.isnan(v[i, j-1])):
                continue

            # Grid spacing in metres (spherical Earth)
            dx = R * np.cos(lat[i, j] * np.pi / 180) * (lon[i, j+1] - lon[i, j-1]) * np.pi / 180
            dy = R * (lat[i+1, j] - lat[i-1, j]) * np.pi / 180

            # Centred finite differences: ∂v/∂x and ∂u/∂y
            dvdx = (v[i, j+1] - v[i, j-1]) / (2 * dx)
            dudy = (u[i+1, j] - u[i-1, j]) / (2 * dy)

            print("i = ", i, " j = ", j)
            print("lat = ", lat[i, j])
            print("lon[i, j+1] = ", lon[i, j+1], " lon[i, j-1] = ", lon[i, j-1])
            print("lat[i+1, j] = ", lat[i+1, j], " lat[i-1, j] = ", lat[i-1, j])
            print("u[i+1, j] = ", u[i+1, j], " u[i-1, j] = ", u[i-1, j])
            print("v[i, j+1] = ", v[i, j+1], " v[i, j-1] = ", v[i, j-1])

            # Relative vorticity: ζ = ∂v/∂x - ∂u/∂y
            vorticity[i, j] = dvdx - dudy

    # Return in the same format as the input
    return vorticity.flatten() if return_1d else vorticity


# --------------------------------------------------------------


def divergence(u, v, lon, lat, nx=None, ny=None):
    """
    Calculates the horizontal divergence (∇·u = ∂u/∂x + ∂v/∂y) using centred
    finite differences on a spherical Earth.
    Accepts both 2D arrays and 1D flattened vectors.

    Parameters:
    -----------
    u, v : ndarray
        East and north velocity components (2D or 1D)
    lon, lat : ndarray
        Longitude and latitude grids (2D or 1D, matching u and v)
    nx : int, optional
        Number of grid points in the x-direction (required for 1D input)
    ny : int, optional
        Number of grid points in the y-direction (required for 1D input)

    Returns:
    --------
    divergence : ndarray
        Divergence field (s⁻¹); same shape as input (returns 1D if input was 1D)
    """
    # If inputs are 1D, reshape to 2D for finite-difference calculations
    if u.ndim == 1:
        if nx is None or ny is None:
            raise ValueError("For 1D vectors, please provide grid dimensions nx and ny")

        u   = u.reshape(ny, nx)
        v   = v.reshape(ny, nx)
        lon = lon.reshape(ny, nx)
        lat = lat.reshape(ny, nx)

        return_1d = True
    else:
        return_1d = False

    ny, nx = u.shape
    divergence = np.full_like(u, np.nan)

    R = 6.371e6  # Earth's radius in metres

    # Central differences on interior grid points (skip boundary rows/columns)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # Skip grid points with any NaN neighbour
            if (np.isnan(u[i, j]) or np.isnan(v[i, j]) or
                    np.isnan(u[i+1, j]) or np.isnan(u[i-1, j]) or
                    np.isnan(v[i, j+1]) or np.isnan(v[i, j-1])):
                continue

            # Grid spacing in metres (spherical Earth)
            dx = R * np.cos(lat[i, j] * np.pi / 180) * (lon[i, j+1] - lon[i, j-1]) * np.pi / 180
            dy = R * (lat[i+1, j] - lat[i-1, j]) * np.pi / 180

            # Centred finite differences: ∂u/∂x and ∂v/∂y
            dudx = (u[i, j+1] - u[i, j-1]) / (2 * dx)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2 * dy)

            # Horizontal divergence: ∇·u = ∂u/∂x + ∂v/∂y
            divergence[i, j] = dudx + dvdy

    # Return in the same format as the input
    return divergence.flatten() if return_1d else divergence


# --------------------------------------------------------------


def kinetic_energy(u, v):
    """
    Calculates the kinetic energy per unit mass: KE = 0.5 * (u² + v²).

    Parameters:
    -----------
    u : ndarray
        East velocity component (m/s)
    v : ndarray
        North velocity component (m/s)

    Returns:
    --------
    ke : ndarray
        Kinetic energy field (m²/s²)
    """
    return 0.5 * (u**2 + v**2)


# --------------------------------------------------------------


def _fill_nan_2d(field):
    """
    Fills NaN gaps in a 2D field by interpolating over the array index grid
    (i, j), so that the FFT used by `kinetic_energy_spectrum` does not
    propagate NaNs to every wavenumber.

    Strategy:
    ---------
    1. Linear interpolation (griddata) using the valid points as support.
    2. Any remaining NaNs (points outside the convex hull of valid data,
       typically along the domain edges) are filled with nearest-neighbour
       interpolation.

    Parameters:
    -----------
    field : ndarray (2D)
        Input field, possibly containing NaNs.

    Returns:
    --------
    filled : ndarray (2D)
        Field with all NaNs replaced by interpolated values. If the field
        has no valid (non-NaN) points at all, it is returned unchanged.
    """
    ny, nx = field.shape
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    valid = ~np.isnan(field)
    if not np.any(valid):
        return field  # nothing to interpolate from

    points = np.column_stack((ii[valid], jj[valid]))
    values = field[valid]
    xi = np.column_stack((ii.ravel(), jj.ravel()))

    filled = griddata(points, values, xi, method='linear').reshape(field.shape)

    still_nan = np.isnan(filled)
    if np.any(still_nan):
        filled_nn = griddata(points, values, xi, method='nearest').reshape(field.shape)
        filled[still_nan] = filled_nn[still_nan]

    return filled


# --------------------------------------------------------------


def kinetic_energy_spectrum(u, v, dx, dy, remove_mean=True, apply_window=True,
                             max_nan_fraction=0.5, crop_to_valid_bbox=True):
    """
    Calculates the isotropic (1D, radially-averaged) kinetic energy spectrum
    of a 2D velocity field on a regular grid, via a 2D FFT.

    Method:
    -------
    0. Cropping to the coverage footprint (`crop_to_valid_bbox`): fields such
       as the HF-radar total-velocity field mapped onto the LS grid are only
       defined within the radar's coverage footprint; every grid node
       outside it is NaN "by construction", not a small gap to fill in. If
       these were left in, they would dominate the NaN fraction and, worse,
       interpolating over them would fabricate data over regions with no
       observation at all. So, by default, the field is first cropped to
       the smallest bounding box containing any valid (non-NaN) point, and
       everything below (NaN fraction check, gap filling, FFT) operates on
       that cropped footprint only.
    1. NaN handling: within the (possibly cropped) field, small remaining
       gaps cannot be passed to an FFT (a single NaN would spread to every
       wavenumber), so they are filled with `_fill_nan_2d`. If the fraction
       of NaNs still exceeds `max_nan_fraction`, a ValueError is raised
       instead of returning a spectrum that would not be trustworthy (e.g.
       a timestep where the radar footprint itself is mostly empty).
    2. The mean (k=0 component) is optionally removed so the spectrum
       describes the turbulent/eddying part of the flow rather than the
       mean current.
    3. A 2D Hann window is optionally applied to taper the field towards
       zero at the domain edges. This is important because the domain is
       not periodic; without tapering, the sharp edges leak energy into
       artificially high wavenumbers (spectral leakage).
    4. u and v are Fourier-transformed (np.fft.fft2) and the 2D kinetic
       energy spectral density is built as
           E_2d(kx, ky) = 0.5 * (|U(kx,ky)|^2 + |V(kx,ky)|^2) * dx * dy / (nx * ny)
       which is the standard periodogram normalisation (so that summing
       E_2d over all (kx, ky) and multiplying by dkx*dky approximates the
       mean kinetic energy of the (detrended) field, i.e. a discrete
       Parseval relation).
    5. The 2D spectrum is azimuthally averaged over rings of constant
       |k| = sqrt(kx^2 + ky^2) to obtain the isotropic 1D spectrum E(k),
       which is what is usually compared against theoretical slopes
       (e.g. k^-3 or k^-5/3) or against another product's spectrum.

    Parameters:
    -----------
    u, v : ndarray (2D)
        East and north velocity components (m/s), on a regular grid.
        May contain NaNs (land, no radar coverage, gaps, etc.).
    dx, dy : float
        Grid spacing in metres along x (longitude) and y (latitude).
    remove_mean : bool
        If True (default), subtract the spatial mean of u and v before
        transforming, so the spectrum reflects the eddying field only.
    apply_window : bool
        If True (default), apply a 2D Hann window before the FFT to reduce
        spectral leakage from the non-periodic domain edges.
    max_nan_fraction : float
        Maximum allowed fraction of NaN points (0-1) *within the cropped
        footprint* (see `crop_to_valid_bbox`). If exceeded, a ValueError is
        raised rather than computing an unreliable spectrum.
    crop_to_valid_bbox : bool
        If True (default), crop the field to the bounding box of valid
        (non-NaN) data before doing anything else. Set to False only if
        `u`/`v` are already a dense field with just a few scattered gaps
        (e.g. a model field with only land points missing).

    Returns:
    --------
    k : ndarray (1D)
        Isotropic wavenumber bins (rad/m), excluding k=0.
    Ek : ndarray (1D)
        Kinetic energy spectral density at each wavenumber (m^3/s^2, i.e.
        energy per unit wavenumber), such that trapz(Ek, k) ≈ mean KE of
        the (detrended) field.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    if u.ndim != 2:
        raise ValueError("u and v must be 2D fields")

    # --- Crop to the bounding box of the actual data coverage -----------
    if crop_to_valid_bbox:
        valid = ~(np.isnan(u) | np.isnan(v))
        if not np.any(valid):
            raise ValueError(
                "kinetic_energy_spectrum: the velocity field has no valid "
                "(non-NaN) points at all."
            )
        rows = np.where(np.any(valid, axis=1))[0]
        cols = np.where(np.any(valid, axis=0))[0]
        u = u[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        v = v[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    ny, nx = u.shape

    # --- Check and fill NaNs -----------------------------------------
    nan_mask = np.isnan(u) | np.isnan(v)
    nan_fraction = np.sum(nan_mask) / nan_mask.size

    if nan_fraction > max_nan_fraction:
        raise ValueError(
            f"kinetic_energy_spectrum: {100 * nan_fraction:.1f}% of the "
            f"velocity field is NaN within its coverage footprint "
            f"(limit={100 * max_nan_fraction:.0f}%); "
            "the spectrum would not be reliable."
        )

    if nan_fraction > 0:
        u = _fill_nan_2d(u)
        v = _fill_nan_2d(v)

    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        raise ValueError(
            "kinetic_energy_spectrum: could not fill all NaNs "
            "(the field may be entirely NaN)."
        )

    # --- Remove mean flow and taper edges ------------------------------
    if remove_mean:
        u = u - np.mean(u)
        v = v - np.mean(v)

    if apply_window:
        window_2d = np.outer(np.hanning(ny), np.hanning(nx))
        # Rescale so the window does not bias the total variance too much
        norm = np.sqrt(np.mean(window_2d ** 2))
        if norm > 0:
            u = u * window_2d / norm
            v = v * window_2d / norm

    # --- 2D FFT and power spectral density ------------------------------
    Lx = nx * dx
    Ly = ny * dy

    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    psd_u = (np.abs(u_hat) ** 2) * dx * dy / (nx * ny)
    psd_v = (np.abs(v_hat) ** 2) * dx * dy / (nx * ny)

    ke_2d = 0.5 * (psd_u + psd_v)

    # --- Wavenumber grid --------------------------------------------------
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mod = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    # --- Azimuthal (radial) averaging into an isotropic 1D spectrum -----
    dkx = 2 * np.pi / Lx
    dky = 2 * np.pi / Ly
    dk = min(dkx, dky)

    k_max = np.max(k_mod)
    n_bins = max(int(np.floor(k_max / dk)), 1)
    k_edges = np.arange(0, n_bins + 1) * dk
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    bin_idx = np.digitize(k_mod.ravel(), k_edges) - 1
    ke_flat = ke_2d.ravel()

    Ek = np.full(len(k_centers), np.nan)
    for b in range(len(k_centers)):
        sel = bin_idx == b
        if np.any(sel):
            # Integrate the 2D PSD over the annulus and divide by dk so
            # that Ek has units of energy per unit wavenumber.
            Ek[b] = np.sum(ke_flat[sel]) * dkx * dky / dk

    # Drop the k=0 bin, it is not meaningful for a spectrum
    valid = k_centers > 0
    return k_centers[valid], Ek[valid]


# --------------------------------------------------------------


def spectral_error(k_ref, Ek_ref, k_test, Ek_test):
    """
    Compares two kinetic-energy spectra (e.g. model vs. observations) and
    quantifies how well the energy distribution across scales agrees,
    independently of comparing the fields point by point in physical space.

    How the error is calculated:
    -----------------------------
    1. The two spectra may come from grids of slightly different size, so
       `Ek_test` is linearly interpolated (np.interp) onto the wavenumber
       axis of the reference spectrum `k_ref`, restricted to the
       overlapping wavenumber range of both spectra.
    2. Per-wavenumber error:
           err_abs(k) = Ek_test(k) - Ek_ref(k)
           err_rel(k) = err_abs(k) / Ek_ref(k)
       `err_abs` shows at which scales (large mesoscale eddies vs. small
       submesoscale features) the energy is over- or under-estimated;
       `err_rel` normalises that by the reference energy at that scale.
    3. A single scalar summary, the spectral error, is computed as the
       relative L2-norm distance between the two spectra:

           spectral_error = || Ek_test - Ek_ref ||_2 / || Ek_ref ||_2
                          = sqrt( sum_k (Ek_test(k) - Ek_ref(k))^2 )
                            / sqrt( sum_k Ek_ref(k)^2 )

       This is the spectral-space analogue of a normalised RMSE: it is 0
       when the two spectra are identical, and grows when energy is
       misplaced across wavenumbers, even if the total (integrated)
       kinetic energy happens to match.

    Parameters:
    -----------
    k_ref, Ek_ref : ndarray (1D)
        Wavenumbers and KE spectral density of the reference field
        (e.g. the model, or the field taken as "truth").
    k_test, Ek_test : ndarray (1D)
        Wavenumbers and KE spectral density of the field being evaluated
        (e.g. the DIVAnd/LS reconstruction).

    Returns:
    --------
    k_common : ndarray (1D)
        Wavenumbers (subset of k_ref) over which the comparison is made.
    err_abs : ndarray (1D)
        Absolute spectral error at each wavenumber (same units as Ek).
    err_rel : ndarray (1D)
        Relative spectral error at each wavenumber (dimensionless).
    spectral_error_scalar : float
        Single relative L2-norm error summarising the whole spectrum.
    """
    k_ref = np.asarray(k_ref, dtype=float)
    Ek_ref = np.asarray(Ek_ref, dtype=float)
    k_test = np.asarray(k_test, dtype=float)
    Ek_test = np.asarray(Ek_test, dtype=float)

    # Keep only the wavenumber range common to both spectra
    k_lo = max(np.nanmin(k_ref), np.nanmin(k_test))
    k_hi = min(np.nanmax(k_ref), np.nanmax(k_test))
    in_range = (k_ref >= k_lo) & (k_ref <= k_hi)

    k_common = k_ref[in_range]
    Ek_ref_common = Ek_ref[in_range]
    Ek_test_common = np.interp(k_common, k_test, Ek_test)

    err_abs = Ek_test_common - Ek_ref_common
    with np.errstate(divide='ignore', invalid='ignore'):
        err_rel = np.where(Ek_ref_common != 0, err_abs / Ek_ref_common, np.nan)

    l2_ref = np.sqrt(np.nansum(Ek_ref_common ** 2))
    spectral_error_scalar = (
        np.sqrt(np.nansum(err_abs ** 2)) / l2_ref if l2_ref > 0 else np.nan
    )

    return k_common, err_abs, err_rel, spectral_error_scalar


# --------------------------------------------------------------


def resample_spectrum(k_target, k_src, Ek_src):
    """
    Linearly resamples a 1D spectrum onto a fixed target wavenumber axis.

    Needed because `kinetic_energy_spectrum` crops each field to its own
    valid-data bounding box (`crop_to_valid_bbox=True`), so the wavenumber
    axis it returns can have a different length/range at every timestep
    (e.g. the HF-radar footprint is larger or smaller depending on
    conditions). To store spectra from many timesteps in a single fixed-size
    array (e.g. a (time, wavenumber) NetCDF variable), each one must first
    be put on a common axis.

    Parameters:
    -----------
    k_target : ndarray (1D)
        The fixed wavenumber axis to resample onto (rad/m).
    k_src, Ek_src : ndarray (1D)
        The wavenumber axis and spectral density actually computed for this
        field (as returned by `kinetic_energy_spectrum` or `spectral_error`).

    Returns:
    --------
    Ek_on_target : ndarray (1D)
        `Ek_src` linearly interpolated onto `k_target`. Values of
        `k_target` outside the range covered by `k_src` are set to NaN
        (extrapolation is not attempted, since it is not meaningful for a
        wavenumber spectrum defined only up to a data-dependent cutoff).
    """
    k_target = np.asarray(k_target, dtype=float)
    k_src = np.asarray(k_src, dtype=float)
    Ek_src = np.asarray(Ek_src, dtype=float)

    return np.interp(k_target, k_src, Ek_src, left=np.nan, right=np.nan)
