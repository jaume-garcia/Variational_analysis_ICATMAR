using PyCall
using CSV, DataFrames
using NCDatasets
using Statistics

# -----------------------------------------------------------------------
# Script: spatial_correlation_phys_var.jl
# Purpose: Computes pointwise TEMPORAL correlation maps (Pearson) between
#          the Copernicus MEDSEA ground truth and three HF radar velocity
#          reconstruction methods (DIVAnd-totals, DIVAnd-radials and Least
#          Squares) for three physical quantities (divergence, vorticity
#          and kinetic energy) over a 264-step time series.
#          This gives 3 quantities x 3 comparisons = 9 correlation maps,
#          each a 2D (lon, lat) field.
#
# Pipeline:
#   1. Load the ICATMAR bathymetry grid, LS grid, Copernicus model, and
#      DIVAnd output fields.
#   2. For each time step:
#      a. Read the LS total velocity file and filter invalid observations.
#      b. Interpolate the Copernicus field to the LS grid.
#      c. Map all velocity fields to the common LS grid using
#         map_total_points_to_LS_grid.
#      d. Compute vorticity, divergence, and kinetic energy for every method
#         and store them in 3D (lon, lat, time) arrays.
#   3. After the loop, compute at every grid point the Pearson correlation
#      along the time dimension between the model and each method
#      (pointwise_temporal_correlation).
#   4. Write all 9 correlation maps to a single NetCDF file.
# -----------------------------------------------------------------------

# --- Julia utility modules ---
include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")
include("../utils/visualization_vel.jl")
include("stats_module.jl")

# --- Add the Python utility directory to sys.path and import modules ---
push!(PyVector(pyimport("sys")."path"), "../utils")

hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")

# Create an instance of HFRadarProcessor (used for reading the LS grid)
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# BATHYMETRY
# ------------------------------------------------------------------------------------

file_icatmar = "../data/bathy.nc"
mask, h, pm, pn, xi, yi = grid_icatmar(file_icatmar)

# ------------------------------------------------------------------------------------
# LEAST SQUARES (LS) OUTPUT GRID
# ------------------------------------------------------------------------------------

grid_file = "../data/hfradar_totals_grid_icatmar.nc"

lon_LS, lat_LS, lon_grid, lat_grid = processor.read_grid_from_netcdf(grid_file)

# Build the full 2D LS coordinate meshgrid
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

# ------------------------------------------------------------------------------------
# COPERNICUS MEDSEA GROUND TRUTH
# ------------------------------------------------------------------------------------

file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"

# Read all time steps at once (@time measures the read time)
@time lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(
    file_cop, "lon", "lat", "u_data", "v_data", "0:"
)

# Build 2D meshgrid and convert from radians to degrees
lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop[:,:], lat_cop[:,:])

lon_mesh_model_cop .= lon_mesh_model_cop .* (180 / pi)
lat_mesh_model_cop .= lat_mesh_model_cop .* (180 / pi)

# Replace fill values (1e20) with NaN
u_model_cop = replace(u_cop, 1.0f20 => NaN32)
v_model_cop = replace(v_cop, 1.0f20 => NaN32)

println("Copernicus dataset successfully loaded")

# ------------------------------------------------------------------------------------
# DIVAND OUTPUT FIELDS
# ------------------------------------------------------------------------------------

# Interpolate the ICATMAR land/sea mask to the LS grid (used for masking later)
mask_interp = ma.interp_grid_eta(
    xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[]
)

divand_file = "../data/january_2026/divand/divand_field_optimal.nc"

# Read all DIVAnd velocity fields in a single NCDataset block
data = NCDataset(divand_file, "r") do ds
    (
        lon      = ds["longitude"][:, :],
        lat      = ds["latitude"][:, :],
        time     = ds["time"][:],
        u_radial = ds["u_radial_divand"][:, :, :],
        v_radial = ds["v_radial_divand"][:, :, :],
        u_total  = ds["u_total_divand"][:, :, :],
        v_total  = ds["v_total_divand"][:, :, :]
    )
end

uri = data.u_radial   # DIVAnd east velocity from radials
vri = data.v_radial   # DIVAnd north velocity from radials
uti = data.u_total    # DIVAnd east velocity from totals
vti = data.v_total    # DIVAnd north velocity from totals

# ------------------------------------------------------------------------------------
# LOCAL FUNCTION: MAP SCATTERED LS POINTS TO THE 2D LS GRID
# ------------------------------------------------------------------------------------

"""
    map_total_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS,
                                u_interp_cop, v_interp_cop, u_array, v_array,
                                uri_timestep, vri_timestep, uti_timestep, vti_timestep)

For each scattered LS observation point, finds the closest node on the LS grid
and assembles four co-located 2D velocity fields (Copernicus, DIVAnd-radials,
DIVAnd-totals, and LS). Also extracts 1D filtered vectors of the co-located
values (NaN points are excluded).

# Arguments
- `lon_array, lat_array`         : Longitudes and latitudes of the LS observation points
- `lon_grid_LS, lat_grid_LS`     : 2D longitude and latitude grids of the LS mesh
- `u_interp_cop, v_interp_cop`   : Copernicus velocity interpolated to the LS grid
- `u_array, v_array`             : LS east and north velocity at observation points
- `uri_timestep, vri_timestep`   : DIVAnd-radials velocity on the LS grid
- `uti_timestep, vti_timestep`   : DIVAnd-totals velocity on the LS grid

# Returns
- `u_model_2d, v_model_2d`       : Copernicus velocity on the 2D LS grid
- `u_rad_2d, v_rad_2d`           : DIVAnd-radials velocity on the 2D LS grid
- `u_tot_2d, v_tot_2d`           : DIVAnd-totals velocity on the 2D LS grid
- `u_LS_2d, v_LS_2d`             : LS velocity on the 2D LS grid
- `lon_array_filtered, lat_array_filtered` : Coordinates of valid comparison points
- `u_interp_cop_filtered, v_interp_cop_filtered` : Copernicus at valid points
- `uri_filtered, vri_filtered`   : DIVAnd-radials at valid points
- `uti_filtered, vti_filtered`   : DIVAnd-totals at valid points
- `u_array_filtered, v_array_filtered`   : LS velocity at valid points
"""
function map_total_points_to_LS_grid(
    lon_array, lat_array,
    lon_grid_LS, lat_grid_LS,
    u_interp_cop, v_interp_cop,
    u_array, v_array,
    uri_timestep, vri_timestep,
    uti_timestep, vti_timestep
)

    tolerance = 1e-6

    # 1D vectors to store co-located values at each LS observation point
    u_interp_cop_at_points = zeros(length(lon_array))
    v_interp_cop_at_points = zeros(length(lon_array))
    uri_at_points = zeros(length(lon_array))
    vri_at_points = zeros(length(lon_array))
    uti_at_points = zeros(length(lon_array))
    vti_at_points = zeros(length(lon_array))

    lon_array_at_points = zeros(length(lon_array))
    lat_array_at_points = zeros(length(lon_array))

    # 2D arrays for the co-located velocity fields (NaN-initialised)
    u_rad_2d = fill(NaN, size(lon_grid_LS))
    v_rad_2d = fill(NaN, size(lon_grid_LS))
    u_tot_2d = fill(NaN, size(lon_grid_LS))
    v_tot_2d = fill(NaN, size(lon_grid_LS))
    u_LS_2d  = fill(NaN, size(lon_grid_LS))
    v_LS_2d  = fill(NaN, size(lon_grid_LS))
    u_model_2d = fill(NaN, size(lon_grid_LS))
    v_model_2d = fill(NaN, size(lon_grid_LS))

    # Counters for diagnostic purposes
    points_found = 0
    points_not_found = 0

    # --- Loop over all LS observation points ---
    for j = 1:length(lon_array)

        # Find the closest LS grid node using Euclidean distance in degree space
        distances = sqrt.(
            (lon_grid_LS .- lon_array[j]).^2 .+
            (lat_grid_LS .- lat_array[j]).^2
        )

        idx_closest = argmin(distances)
        idx_lon, idx_lat = Tuple(idx_closest)

        # Verify whether the closest node coincides with the observation point
        lon_closest = lon_grid_LS[idx_lon, idx_lat]
        lat_closest = lat_grid_LS[idx_lon, idx_lat]

        is_grid_point =
            abs(lon_closest - lon_array[j]) < tolerance &&
            abs(lat_closest - lat_array[j]) < tolerance

        if is_grid_point
            points_found += 1
        else
            points_not_found += 1
        end

        # Assign velocity values to the 2D arrays at the closest grid node
        u_model_2d[idx_lon, idx_lat] = u_interp_cop[idx_lon, idx_lat]
        v_model_2d[idx_lon, idx_lat] = v_interp_cop[idx_lon, idx_lat]

        u_rad_2d[idx_lon, idx_lat] = uri_timestep[idx_lon, idx_lat]
        v_rad_2d[idx_lon, idx_lat] = vri_timestep[idx_lon, idx_lat]

        u_tot_2d[idx_lon, idx_lat] = uti_timestep[idx_lon, idx_lat]
        v_tot_2d[idx_lon, idx_lat] = vti_timestep[idx_lon, idx_lat]

        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]

        # Extract co-located values into 1D vectors
        u_interp_cop_at_points[j] = u_interp_cop[idx_lon, idx_lat]
        v_interp_cop_at_points[j] = v_interp_cop[idx_lon, idx_lat]
        uri_at_points[j] = uri_timestep[idx_lon, idx_lat]
        vri_at_points[j] = vri_timestep[idx_lon, idx_lat]
        uti_at_points[j] = uti_timestep[idx_lon, idx_lat]
        vti_at_points[j] = vti_timestep[idx_lon, idx_lat]

        lon_array_at_points[j] = lon_closest
        lat_array_at_points[j] = lat_closest
    end

    # Build a mask for points that are valid (non-NaN) in all methods simultaneously
    valid_mask = .!(
        isnan.(u_interp_cop_at_points) .|
        isnan.(v_interp_cop_at_points) .|
        isnan.(uri_at_points) .|
        isnan.(vri_at_points) .|
        isnan.(uti_at_points) .|
        isnan.(vti_at_points) .|
        isnan.(u_array) .|
        isnan.(v_array)
    )

    # Return the 2D fields plus only the commonly valid co-located points
    return u_model_2d, v_model_2d,
           u_rad_2d, v_rad_2d,
           u_tot_2d, v_tot_2d,
           u_LS_2d, v_LS_2d,
           lon_array_at_points[valid_mask],
           lat_array_at_points[valid_mask],
           u_interp_cop_at_points[valid_mask],
           v_interp_cop_at_points[valid_mask],
           uri_at_points[valid_mask],
           vri_at_points[valid_mask],
           uti_at_points[valid_mask],
           vti_at_points[valid_mask],
           u_array[valid_mask],
           v_array[valid_mask]

end

# ------------------------------------------------------------------------------------
# LOCAL FUNCTION: POINTWISE TEMPORAL CORRELATION
# ------------------------------------------------------------------------------------

"""
    pointwise_temporal_correlation(field_a, field_b; min_valid=3)

For two 3D arrays of size (nlon, nlat, ntime), computes the Pearson correlation
coefficient along the time dimension at every grid point, using only the
timesteps where BOTH series are valid (not NaN) at that point. This is needed
because radar coverage (and therefore data availability) varies from timestep
to timestep.

# Arguments
- `field_a, field_b` : 3D arrays (nlon, nlat, ntime) with identical size
- `min_valid`        : Minimum number of common valid timesteps required to
                       compute a correlation (default: 3)

# Returns
- `corr_map` : 2D array (nlon, nlat) of correlation coefficients. Points with
               fewer than `min_valid` common valid timesteps, or with zero
               variance in either series, are left as NaN.
"""
function pointwise_temporal_correlation(field_a::Array{Float64,3},
                                         field_b::Array{Float64,3};
                                         min_valid::Int = 3)

    @assert size(field_a) == size(field_b) "field_a and field_b must have the same size"

    nlon, nlat, ntime = size(field_a)
    corr_map = fill(NaN, nlon, nlat)

    for ilon = 1:nlon, ilat = 1:nlat

        # Time series of both fields at this grid point (views avoid copies)
        a_series = @view field_a[ilon, ilat, :]
        b_series = @view field_b[ilon, ilat, :]

        # Timesteps where both series are valid
        valid = .!(isnan.(a_series) .| isnan.(b_series))

        if count(valid) >= min_valid
            a_valid = a_series[valid]
            b_valid = b_series[valid]

            # Correlation is undefined if either series is constant
            if std(a_valid) > 0 && std(b_valid) > 0
                corr_map[ilon, ilat] = cor(a_valid, b_valid)
            end
        end
    end

    return corr_map
end

# ------------------------------------------------------------------------------------
# MAIN LOOP: COMPUTE PHYSICAL QUANTITIES FOR EACH HOURLY SNAPSHOT
# ------------------------------------------------------------------------------------

n_times = 264
nlon, nlat = size(lon_grid_LS)

# 3D storage (nlon, nlat, n_times) for the full time series of each physical
# quantity at every grid point, for the model and for each of the three methods
# it is compared against (DIVAnd totals, DIVAnd radials, LS totals). The pointwise
# correlation is computed from these once the loop finishes.
divergence_model_ts = fill(NaN, nlon, nlat, n_times)
divergence_tot_ts   = fill(NaN, nlon, nlat, n_times)
divergence_rad_ts   = fill(NaN, nlon, nlat, n_times)
divergence_LS_ts    = fill(NaN, nlon, nlat, n_times)

vorticity_model_ts  = fill(NaN, nlon, nlat, n_times)
vorticity_tot_ts    = fill(NaN, nlon, nlat, n_times)
vorticity_rad_ts    = fill(NaN, nlon, nlat, n_times)
vorticity_LS_ts     = fill(NaN, nlon, nlat, n_times)

ke_model_ts         = fill(NaN, nlon, nlat, n_times)
ke_tot_ts           = fill(NaN, nlon, nlat, n_times)
ke_rad_ts           = fill(NaN, nlon, nlat, n_times)
ke_LS_ts            = fill(NaN, nlon, nlat, n_times)

for i = 1:n_times

    println("Processing timestep: ", i)

    # ── Copernicus snapshot at time step i ────────────────────────────
    u_cop_timestep = u_model_cop[i,:,:]
    v_cop_timestep = v_model_cop[i,:,:]

    # ── DIVAnd velocity snapshots at time step i ─────────────────────
    uri_timestep = uri[i,:,:]   # Radials – east component
    vri_timestep = vri[i,:,:]   # Radials – north component
    uti_timestep = uti[i,:,:]   # Totals  – east component
    vti_timestep = vti[i,:,:]   # Totals  – north component

    # ── Least Squares total velocities for snapshot i ────────────────
    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_" *
               lpad(string(i-1), 3, '0') * "_all_grid.txt"

    df = CSV.read(file_tuv, DataFrame;
        delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array = df.u_total      # East velocity (m/s)
    v_array = df.v_total      # North velocity (m/s)

    # Filter out geometrically poor (GDOP > 2) and physically unrealistic (|v| > 1.2 m/s) points
    valid_index = (df.gdop .<= 2.0) .& (df.speed .<= 1.2)

    lon_array = lon_array[valid_index]
    lat_array = lat_array[valid_index]
    u_array = u_array[valid_index]
    v_array = v_array[valid_index]

    # ── Interpolate Copernicus to the LS grid ────────────────────────
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(
        lon_mesh_model_cop, lat_mesh_model_cop,
        u_cop_timestep, v_cop_timestep,
        lon_grid_LS, lat_grid_LS,
        mask_coarse_grid=[], points_mode=false
    )

    # ── Map all fields to the common LS grid ─────────────────────────
    u_model_2d, v_model_2d,
    u_rad_2d, v_rad_2d,
    u_tot_2d, v_tot_2d,
    u_LS_2d, v_LS_2d,
    lon_array_filtered, lat_array_filtered,
    u_interp_cop_filtered, v_interp_cop_filtered,
    uri_filtered, vri_filtered,
    uti_filtered, vti_filtered,
    u_array_filtered, v_array_filtered =
        map_total_points_to_LS_grid(
            lon_array, lat_array,
            lon_grid_LS, lat_grid_LS,
            u_interp_cop, v_interp_cop,
            u_array, v_array,
            uri_timestep, vri_timestep,
            uti_timestep, vti_timestep
        )

    # ── Vorticity (∂v/∂x − ∂u/∂y), divergence (∂u/∂x + ∂v/∂y) and
    #    kinetic energy (KE = 0.5*(u² + v²)) for each method, stored at time step i ──

    # Copernicus model (ground truth)
    vorticity_model_ts[:, :, i]  = calculate_vorticity(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    divergence_model_ts[:, :, i] = calculate_divergence(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    ke_model_ts[:, :, i]         = calculate_kinetic_energy(u_model_2d, v_model_2d)

    # DIVAnd totals
    vorticity_tot_ts[:, :, i]  = calculate_vorticity(u_tot_2d, v_tot_2d, lon_grid_LS, lat_grid_LS)
    divergence_tot_ts[:, :, i] = calculate_divergence(u_tot_2d, v_tot_2d, lon_grid_LS, lat_grid_LS)
    ke_tot_ts[:, :, i]         = calculate_kinetic_energy(u_tot_2d, v_tot_2d)

    # DIVAnd radials
    vorticity_rad_ts[:, :, i]  = calculate_vorticity(u_rad_2d, v_rad_2d, lon_grid_LS, lat_grid_LS)
    divergence_rad_ts[:, :, i] = calculate_divergence(u_rad_2d, v_rad_2d, lon_grid_LS, lat_grid_LS)
    ke_rad_ts[:, :, i]         = calculate_kinetic_energy(u_rad_2d, v_rad_2d)

    # Least Squares totals
    vorticity_LS_ts[:, :, i]  = calculate_vorticity(u_LS_2d, v_LS_2d, lon_grid_LS, lat_grid_LS)
    divergence_LS_ts[:, :, i] = calculate_divergence(u_LS_2d, v_LS_2d, lon_grid_LS, lat_grid_LS)
    ke_LS_ts[:, :, i]         = calculate_kinetic_energy(u_LS_2d, v_LS_2d)

end

# ------------------------------------------------------------------------------------
# POINTWISE TEMPORAL CORRELATION MAPS (METHOD vs MODEL)
# ------------------------------------------------------------------------------------

println("\nComputing pointwise temporal correlation maps...")

# --- DIVAnd total field vs model ---
corr_divergence_tot = pointwise_temporal_correlation(divergence_model_ts, divergence_tot_ts)
corr_vorticity_tot  = pointwise_temporal_correlation(vorticity_model_ts, vorticity_tot_ts)
corr_ke_tot         = pointwise_temporal_correlation(ke_model_ts, ke_tot_ts)

# --- DIVAnd radial field vs model ---
corr_divergence_rad = pointwise_temporal_correlation(divergence_model_ts, divergence_rad_ts)
corr_vorticity_rad  = pointwise_temporal_correlation(vorticity_model_ts, vorticity_rad_ts)
corr_ke_rad         = pointwise_temporal_correlation(ke_model_ts, ke_rad_ts)

# --- LS total field vs model ---
corr_divergence_LS = pointwise_temporal_correlation(divergence_model_ts, divergence_LS_ts)
corr_vorticity_LS  = pointwise_temporal_correlation(vorticity_model_ts, vorticity_LS_ts)
corr_ke_LS         = pointwise_temporal_correlation(ke_model_ts, ke_LS_ts)

# ------------------------------------------------------------------------------------
# SAVE CORRELATION MAPS TO NETCDF
# ------------------------------------------------------------------------------------

output_dir = "../data/january_2026/stats_data/"
isdir(output_dir) || mkpath(output_dir)

output_file = joinpath(output_dir, "correlation_maps_icatmar_optimal.nc")

# Remove any previous version of the output file
isfile(output_file) && rm(output_file)

NCDataset(output_file, "c") do ds

    defDim(ds, "lon", nlon)
    defDim(ds, "lat", nlat)

    # --- Coordinates ---
    v_lon = defVar(ds, "longitude", Float64, ("lon", "lat"))
    v_lon[:, :] = lon_grid_LS
    v_lon.attrib["units"] = "degrees_east"
    v_lon.attrib["long_name"] = "Longitude"

    v_lat = defVar(ds, "latitude", Float64, ("lon", "lat"))
    v_lat[:, :] = lat_grid_LS
    v_lat.attrib["units"] = "degrees_north"
    v_lat.attrib["long_name"] = "Latitude"

    # --- DIVAnd TOTAL FIELD vs MODEL ---
    v_corr_div_tot = defVar(ds, "correlation_divergence_tot", Float64, ("lon", "lat"))
    v_corr_div_tot[:, :] = corr_divergence_tot
    v_corr_div_tot.attrib["units"] = "1"
    v_corr_div_tot.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the DIVAnd total field divergence, at each grid point"

    v_corr_vort_tot = defVar(ds, "correlation_vorticity_tot", Float64, ("lon", "lat"))
    v_corr_vort_tot[:, :] = corr_vorticity_tot
    v_corr_vort_tot.attrib["units"] = "1"
    v_corr_vort_tot.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the DIVAnd total field vorticity, at each grid point"

    v_corr_ke_tot = defVar(ds, "correlation_kinetic_energy_tot", Float64, ("lon", "lat"))
    v_corr_ke_tot[:, :] = corr_ke_tot
    v_corr_ke_tot.attrib["units"] = "1"
    v_corr_ke_tot.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the DIVAnd total field kinetic energy, at each grid point"

    # --- DIVAnd RADIAL FIELD vs MODEL ---
    v_corr_div_rad = defVar(ds, "correlation_divergence_rad", Float64, ("lon", "lat"))
    v_corr_div_rad[:, :] = corr_divergence_rad
    v_corr_div_rad.attrib["units"] = "1"
    v_corr_div_rad.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the DIVAnd radial field divergence, at each grid point"

    v_corr_vort_rad = defVar(ds, "correlation_vorticity_rad", Float64, ("lon", "lat"))
    v_corr_vort_rad[:, :] = corr_vorticity_rad
    v_corr_vort_rad.attrib["units"] = "1"
    v_corr_vort_rad.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the DIVAnd radial field vorticity, at each grid point"

    v_corr_ke_rad = defVar(ds, "correlation_kinetic_energy_rad", Float64, ("lon", "lat"))
    v_corr_ke_rad[:, :] = corr_ke_rad
    v_corr_ke_rad.attrib["units"] = "1"
    v_corr_ke_rad.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the DIVAnd radial field kinetic energy, at each grid point"

    # --- LS TOTAL FIELD vs MODEL ---
    v_corr_div_LS = defVar(ds, "correlation_divergence_LS", Float64, ("lon", "lat"))
    v_corr_div_LS[:, :] = corr_divergence_LS
    v_corr_div_LS.attrib["units"] = "1"
    v_corr_div_LS.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the LS total field divergence, at each grid point"

    v_corr_vort_LS = defVar(ds, "correlation_vorticity_LS", Float64, ("lon", "lat"))
    v_corr_vort_LS[:, :] = corr_vorticity_LS
    v_corr_vort_LS.attrib["units"] = "1"
    v_corr_vort_LS.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the LS total field vorticity, at each grid point"

    v_corr_ke_LS = defVar(ds, "correlation_kinetic_energy_LS", Float64, ("lon", "lat"))
    v_corr_ke_LS[:, :] = corr_ke_LS
    v_corr_ke_LS.attrib["units"] = "1"
    v_corr_ke_LS.attrib["long_name"] =
        "Temporal Pearson correlation coefficient between the Copernicus model " *
        "and the LS total field kinetic energy, at each grid point"

    # --- Global attributes ---
    ds.attrib["title"] =
        "Pointwise temporal correlation maps: model vs. DIVAnd total, DIVAnd radial and LS total fields"
    ds.attrib["source_n_times"] = n_times
end

println("NetCDF file generated: ", output_file)