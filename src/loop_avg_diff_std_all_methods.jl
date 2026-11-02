using PyCall
using PyPlot
using CSV, DataFrames
using DelimitedFiles
using NCDatasets
using NaNMath

# -----------------------------------------------------------------------
# Script: loop_avg_diff_std_all_methods.jl
# Purpose: Compares three HF radar velocity reconstruction methods
#          (DIVAnd-radials, DIVAnd-totals and Least Squares) against the
#          Copernicus MEDSEA ground truth in terms of velocity magnitude
#          over a 264-step time series. For every method it computes:
#            - the domain-averaged velocity magnitude at each time step,
#            - the difference with MEDSEA (u, v and magnitude) and the
#              relative velocity difference,
#            - the temporal mean and temporal standard deviation of the
#              differences at every grid point.
#
# Pipeline:
#   1. Load the ICATMAR bathymetry grid, LS grid, Copernicus model, and
#      DIVAnd output fields.
#   2. For each time step:
#      a. Read the LS total velocity file and filter invalid observations.
#      b. Place the LS points on the LS grid and interpolate the Copernicus
#         field to the LS grid.
#      c. Keep only the grid points valid (non-NaN) in all methods.
#      d. Compute velocity magnitudes, their domain average, the
#         differences with MEDSEA and the relative differences.
#      e. Compute the average of the spatial standard deviation of the
#         magnitude differences.
#   3. Compute the temporal mean and temporal standard deviation of the
#      differences at every grid point.
#   4. Write the time series to ASCII files, and the mean and standard
#      deviation maps to NetCDF files.
# -----------------------------------------------------------------------

# --- Julia utility modules ---
include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")
include("../utils/visualization_vel.jl")

# --- Add the Python utility directory to sys.path and import modules ---
push!(PyVector(pyimport("sys")."path"), "../utils")

hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")

# Create an instance of HFRadarProcessor (used for reading the LS grid)
processor = hf.HFRadarProcessor()

# ------------------------------------------------------------------------------------
# LOCAL FUNCTIONS
# ------------------------------------------------------------------------------------

"""
    nanstd(arr, dim)

Standard deviation of `arr` along dimension `dim`, ignoring NaN values.
The mean is computed over the non-NaN values, and the variance is the mean
of the squared deviations over the non-NaN values (population standard
deviation, i.e. normalised by N).

# Arguments
- `arr` : Array of any dimension that may contain NaNs
- `dim` : Dimension along which the standard deviation is computed

# Returns
- Array with dimension `dim` removed, containing the standard deviation
  (NaN where there are no valid values).
"""
function nanstd(arr, dim)

    # Mean ignoring NaN
    μ = dropdims(sum(x -> isnan(x) ? 0.0 : x, arr, dims=dim) ./ 
                 sum(x -> isnan(x) ? 0 : 1, arr, dims=dim), dims=dim)
    
    # Expand the mean so it can be broadcast against the original array
    μ_expanded = reshape(μ, ntuple(i -> i == dim ? 1 : size(μ, i > dim ? i-1 : i), ndims(arr)))
    
    # Squared differences
    squared_diff = (arr .- μ_expanded).^2
    
    # Variance ignoring NaN
    variance = dropdims(sum(x -> isnan(x) ? 0.0 : x, squared_diff, dims=dim) ./ 
                        sum(x -> isnan(x) ? 0 : 1, squared_diff, dims=dim), dims=dim)
    
    return sqrt.(variance)

end

"""
    prova_nanstd(arr, dim)

Same as `nanstd`, but if `arr` contains NaNs the result is returned as a 1D
vector with the NaN entries (positions with no valid data) removed. This
allows taking a plain `mean` of the result afterwards. If `arr` has no NaNs,
the full standard deviation array is returned.

# Arguments
- `arr` : Array of any dimension that may contain NaNs
- `dim` : Dimension along which the standard deviation is computed

# Returns
- Standard deviation array (no NaNs) or the NaN-free 1D vector described above.
"""
function prova_nanstd(arr, dim)

    has_nans = any(isnan, arr)
    
    # Mean ignoring NaN
    μ = dropdims(sum(x -> isnan(x) ? 0.0 : x, arr, dims=dim) ./ 
                 sum(x -> isnan(x) ? 0 : 1, arr, dims=dim), dims=dim)
    
    # Expand the mean so it can be broadcast against the original array
    μ_expanded = reshape(μ, ntuple(i -> i == dim ? 1 : size(μ, i > dim ? i-1 : i), ndims(arr)))
    
    # Squared differences
    squared_diff = (arr .- μ_expanded).^2
    
    # Variance ignoring NaN
    variance = dropdims(sum(x -> isnan(x) ? 0.0 : x, squared_diff, dims=dim) ./ 
                        sum(x -> isnan(x) ? 0 : 1, squared_diff, dims=dim), dims=dim)
    
    # Drop the positions without valid data if the input contained NaNs
    if has_nans
        valid_mask = .!(isnan.(variance))
        variance_clean = variance[valid_mask]
        return sqrt.(variance_clean)
    else
        return sqrt.(variance)
    end
end

# ------------------------------------------------------------------------------------
# BATHYMETRY
# ------------------------------------------------------------------------------------

file_icatmar = "../data/bathy.nc"

mask, h, pm, pn, xi, yi = grid_icatmar(file_icatmar);

# ------------------------------------------------------------------------------------
# LEAST SQUARES (LS) OUTPUT GRID
# ------------------------------------------------------------------------------------

grid_file = "../data/hfradar_totals_grid_icatmar.nc"

lon_LS, lat_LS, lon_grid, lat_grid, mask_antena = processor.read_grid_from_netcdf(grid_file)

# Build the full 2D LS coordinate meshgrid
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)

# ------------------------------------------------------------------------------------
# COPERNICUS MEDSEA GROUND TRUTH
# ------------------------------------------------------------------------------------

file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"

# Read all time steps at once
lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(file_cop, "lon", "lat", "u_data", "v_data", "0:")

# Build 2D meshgrid and convert from radians to degrees
lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop[:, :], lat_cop[:, :])

lon_mesh_model_cop = lon_mesh_model_cop .* (180 / pi)
lat_mesh_model_cop = lat_mesh_model_cop .* (180 / pi)

# Replace fill values (1e20) with NaN
u_model_cop = replace(u_cop, 1.0f20 => NaN32)
v_model_cop = replace(v_cop, 1.0f20 => NaN32)

println("COPERNICUS read")

# ------------------------------------------------------------------------------------
# DIVAND OUTPUT FIELDS
# ------------------------------------------------------------------------------------

# Interpolate the ICATMAR land/sea mask to the LS grid
mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

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
# MAIN LOOP: COMPUTE STATISTICS FOR EACH HOURLY SNAPSHOT
# ------------------------------------------------------------------------------------

n_times   = 264
grid_size = size(uri[1, :, :])

# Accumulators for the differences with MEDSEA (method - MEDSEA) of all time steps:
# east component (u), north component (v) and velocity magnitude (mag)
diff_rad_u_all   = Array{Float32}(undef, n_times, grid_size...)
diff_rad_v_all   = Array{Float32}(undef, n_times, grid_size...)
diff_rad_mag_all = Array{Float32}(undef, n_times, grid_size...)

diff_tot_u_all   = Array{Float32}(undef, n_times, grid_size...)
diff_tot_v_all   = Array{Float32}(undef, n_times, grid_size...)
diff_tot_mag_all = Array{Float32}(undef, n_times, grid_size...)

diff_LS_u_all    = Array{Float32}(undef, n_times, grid_size...)
diff_LS_v_all    = Array{Float32}(undef, n_times, grid_size...)
diff_LS_mag_all  = Array{Float32}(undef, n_times, grid_size...)

# Accumulators for the relative velocity magnitude difference
# (method - MEDSEA) / MEDSEA of all time steps
vel_rel_rad_mag_all = Array{Float32}(undef, n_times, grid_size...)
vel_rel_tot_mag_all = Array{Float32}(undef, n_times, grid_size...)
vel_rel_LS_mag_all  = Array{Float32}(undef, n_times, grid_size...)

# Number of valid points per time step (for radials and totals)
diff_n_points_valid_rad = zeros(n_times, 1)
diff_n_points_valid_tot = zeros(n_times, 1)

# Time series (one row per time step, 5 columns):
#   vel_time_series: time index, domain-mean velocity magnitude (MEDSEA, radials, totals, LS)
#   std_time_series: time index, number of valid points, mean std of the magnitude
#                    difference (radials, totals, LS)
vel_time_series = zeros(n_times, 5)
std_time_series = zeros(n_times, 5)

for i = 1:n_times

    println("ITERATION ... ", i)

    # ── Copernicus snapshot at time step i ────────────────────────────
    u_cop_timestep = u_model_cop[i, :, :]
    v_cop_timestep = v_model_cop[i, :, :]
    
    # ── DIVAnd velocity snapshots at time step i ─────────────────────
    uri_timestep = uri[i, :, :]   # Radials – east component
    vri_timestep = vri[i, :, :]   # Radials – north component
    uti_timestep = uti[i, :, :]   # Totals  – east component
    vti_timestep = vti[i, :, :]   # Totals  – north component
    
    # ── Least Squares total velocities (L3) for snapshot i ───────────
    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_" * lpad(string(i-1), 3, '0') * "_all_grid.txt"

    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array     = df.longitude
    lat_array     = df.latitude
    u_array       = (df.u_total)      # East velocity (m/s)
    v_array       = (df.v_total)      # North velocity (m/s)
    mod_vel_array = (df.speed)        # Velocity magnitude (m/s)
    angle_array   = df.direction      # Velocity direction
    gdop_array    = df.gdop           # Geometric Dilution of Precision

    # Filter out geometrically poor (GDOP > 2) and physically unrealistic (|v| > 1.2 m/s) points
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)

    lon_array     = lon_array[valid_index]
    lat_array     = lat_array[valid_index]
    u_array       = u_array[valid_index]
    v_array       = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array   = angle_array[valid_index]
    
    println("TOTALS COP read");
    
    # ── Place the LS points on the 2D LS grid ────────────────────────
    u_LS_2d, v_LS_2d = ma.putting_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS, u_array, v_array)
    
    # ── Interpolate Copernicus to the LS grid ────────────────────────
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(lon_mesh_model_cop, lat_mesh_model_cop, u_cop_timestep, v_cop_timestep, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[], points_mode=false)
    
    # ── Mask of grid points valid (non-NaN) in all methods simultaneously ──
    valid_mask = .!(isnan.(u_interp_cop) .| isnan.(v_interp_cop) .|
                isnan.(uri_timestep) .| isnan.(vri_timestep) .|
                isnan.(uti_timestep) .| isnan.(vti_timestep) .|
                isnan.(u_LS_2d) .| isnan.(v_LS_2d))
                
    # Extract only the commonly valid points for comparisons
    u_interp_cop_filtered = u_interp_cop[valid_mask]
    v_interp_cop_filtered = v_interp_cop[valid_mask]
    uri_filtered          = uri_timestep[valid_mask]
    vri_filtered          = vri_timestep[valid_mask]
    uti_filtered          = uti_timestep[valid_mask]
    vti_filtered          = vti_timestep[valid_mask]
    u_array_filtered      = u_LS_2d[valid_mask]
    v_array_filtered      = v_LS_2d[valid_mask]
    
    # ── Velocity magnitude (sqrt(u² + v²)) for each method ───────────
    vel_mag_cop = sqrt.(u_interp_cop_filtered.^2 .+ v_interp_cop_filtered.^2)
    vel_mag_rad = sqrt.(uri_filtered.^2 .+ vri_filtered.^2)
    vel_mag_tot = sqrt.(uti_filtered.^2 .+ vti_filtered.^2)
    vel_mag_LS  = sqrt.(u_array_filtered.^2 .+ v_array_filtered.^2)
    
    # ── Domain-averaged velocity magnitude ───────────────────────────
    avg_vel_cop = mean(vel_mag_cop)
    avg_vel_rad = mean(vel_mag_rad)
    avg_vel_tot = mean(vel_mag_tot)
    avg_vel_LS  = mean(vel_mag_LS)
    
    println("AVERAGES:")
    println("COP = ", avg_vel_cop)
    println("RAD = ", avg_vel_rad)
    println("TOT = ", avg_vel_tot)
    println("LS = ", avg_vel_LS)
    
    vel_time_series[i, :] = [i, avg_vel_cop, avg_vel_rad, avg_vel_tot, avg_vel_LS]
    
    # ── Differences with MEDSEA (2D fields initialised with NaN) ─────
    diff_rad_u   = fill(NaN32, grid_size...)
    diff_rad_v   = fill(NaN32, grid_size...)
    diff_rad_mag = fill(NaN32, grid_size...)

    diff_tot_u   = fill(NaN32, grid_size...)
    diff_tot_v   = fill(NaN32, grid_size...)
    diff_tot_mag = fill(NaN32, grid_size...)

    diff_LS_u    = fill(NaN32, grid_size...)
    diff_LS_v    = fill(NaN32, grid_size...)
    diff_LS_mag  = fill(NaN32, grid_size...)
    
    vel_relative_rad = fill(NaN32, grid_size...)
    vel_relative_tot = fill(NaN32, grid_size...)
    vel_relative_LS  = fill(NaN32, grid_size...)

    # Differences (method - MEDSEA) for the components and the magnitude
    diff_rad_u[valid_mask]   = uri_filtered .- u_interp_cop_filtered
    diff_rad_v[valid_mask]   = vri_filtered .- v_interp_cop_filtered
    diff_rad_mag[valid_mask] = vel_mag_rad .- vel_mag_cop

    diff_tot_u[valid_mask]   = uti_filtered .- u_interp_cop_filtered
    diff_tot_v[valid_mask]   = vti_filtered .- v_interp_cop_filtered
    diff_tot_mag[valid_mask] = vel_mag_tot .- vel_mag_cop

    diff_LS_u[valid_mask]    = u_array_filtered .- u_interp_cop_filtered
    diff_LS_v[valid_mask]    = v_array_filtered .- v_interp_cop_filtered
    diff_LS_mag[valid_mask]  = vel_mag_LS .- vel_mag_cop
    
    # Relative velocity magnitude difference: (method - MEDSEA) / MEDSEA
    vel_relative_rad[valid_mask] = (vel_mag_rad .- vel_mag_cop) ./ vel_mag_cop
    vel_relative_tot[valid_mask] = (vel_mag_tot .- vel_mag_cop) ./ vel_mag_cop
    vel_relative_LS[valid_mask]  = (vel_mag_LS .- vel_mag_cop) ./ vel_mag_cop

    # ── Store the differences of this time step ──────────────────────
    diff_rad_u_all[i, :, :]   = diff_rad_u
    diff_rad_v_all[i, :, :]   = diff_rad_v
    diff_rad_mag_all[i, :, :] = diff_rad_mag

    diff_tot_u_all[i, :, :]   = diff_tot_u
    diff_tot_v_all[i, :, :]   = diff_tot_v
    diff_tot_mag_all[i, :, :] = diff_tot_mag

    diff_LS_u_all[i, :, :]    = diff_LS_u
    diff_LS_v_all[i, :, :]    = diff_LS_v
    diff_LS_mag_all[i, :, :]  = diff_LS_mag
    
    # ── Store the relative differences of this time step ─────────────
    vel_rel_rad_mag_all[i, :, :] = vel_relative_rad
    vel_rel_tot_mag_all[i, :, :] = vel_relative_tot
    vel_rel_LS_mag_all[i, :, :]  = vel_relative_LS
        
    # ── Average of the standard deviation of the magnitude difference ──
    avg_std_rad = mean(prova_nanstd(diff_rad_mag_all[i, :, :], 1))
    avg_std_tot = mean(prova_nanstd(diff_tot_mag_all[i, :, :], 1))
    avg_std_LS  = mean(prova_nanstd(diff_LS_mag_all[i, :, :], 1))
    
    println("STANDARD DEVIATION AVERAGES: ...")
    println("RAD = ... ", avg_std_rad)
    println("TOT = ... ", avg_std_tot)
    println("LS = ... ", avg_std_LS)
    
    # Number of grid points valid in all methods at this time step
    n_valid_points = sum(valid_mask)

    println("Puntos válidos en esta iteración: ", n_valid_points)
    
    std_time_series[i, :] = [i, n_valid_points, avg_std_rad, avg_std_tot, avg_std_LS]
    
end

# ------------------------------------------------------------------------------------
# TEMPORAL MEAN OF THE DIFFERENCES
# ------------------------------------------------------------------------------------

"""
    avg_temp(diff_vel)

Temporal mean (along the first dimension, time) of a 3D array (time, x, y),
ignoring NaN values. At every grid point the mean is taken over the time
steps where the data are valid.

# Arguments
- `diff_vel` : 3D array (n_times, nx, ny) that may contain NaNs

# Returns
- 2D array (nx, ny) with the NaN-ignoring temporal mean.
"""
function avg_temp(diff_vel)
    
    return dropdims(mean(x -> isnan(x) ? zero(x) : x, diff_vel, dims=1), dims=1) ./ 
                  dropdims(mean(x -> isnan(x) ? 0 : 1, diff_vel, dims=1), dims=1)
end

# Temporal mean of the differences with MEDSEA
diff_rad_u_mean   = avg_temp(diff_rad_u_all)
diff_rad_v_mean   = avg_temp(diff_rad_v_all)
diff_rad_mag_mean = avg_temp(diff_rad_mag_all)

diff_tot_u_mean   = avg_temp(diff_tot_u_all)
diff_tot_v_mean   = avg_temp(diff_tot_v_all)
diff_tot_mag_mean = avg_temp(diff_tot_mag_all)

diff_LS_u_mean    = avg_temp(diff_LS_u_all)
diff_LS_v_mean    = avg_temp(diff_LS_v_all)
diff_LS_mag_mean  = avg_temp(diff_LS_mag_all)

# Temporal mean of the relative velocity differences
vel_rel_rad_mag_mean = avg_temp(vel_rel_rad_mag_all)
vel_rel_tot_mag_mean = avg_temp(vel_rel_tot_mag_all)
vel_rel_LS_mag_mean  = avg_temp(vel_rel_LS_mag_all)

# ------------------------------------------------------------------------------------
# SAVE TIME SERIES TO ASCII FILES
# ------------------------------------------------------------------------------------

# Domain-averaged velocity magnitude time series
output_file = "../data/january_2026/stats_data/velocity_magnitude_times_series.txt"
writedlm(output_file, vel_time_series, ' ')
println("Archivo VEL generado: ", output_file)

# Standard deviation of the differences time series
output_file = "../data/january_2026/stats_data/standard_deviation_diff_times_series.txt"
writedlm(output_file, std_time_series, ' ')
println("Archivo STD generado: ", output_file)

# ------------------------------------------------------------------------------------
# TEMPORAL STANDARD DEVIATION OF THE DIFFERENCES
# ------------------------------------------------------------------------------------

# Standard deviation along time at every grid point
std_diff_rad_u   = nanstd(diff_rad_u_all, 1) 
std_diff_rad_v   = nanstd(diff_rad_v_all, 1)
std_diff_rad_mag = nanstd(diff_rad_mag_all, 1) 

std_diff_tot_u   = nanstd(diff_tot_u_all, 1) 
std_diff_tot_v   = nanstd(diff_tot_v_all, 1) 
std_diff_tot_mag = nanstd(diff_tot_mag_all, 1) 

std_diff_LS_u    = nanstd(diff_LS_u_all, 1) 
std_diff_LS_v    = nanstd(diff_LS_v_all, 1) 
std_diff_LS_mag  = nanstd(diff_LS_mag_all, 1)

# Points with a zero standard deviation have no (or a single) valid sample: set them to NaN
std_diff_rad_u[std_diff_rad_u .== 0]     .= NaN
std_diff_rad_v[std_diff_rad_v .== 0]     .= NaN
std_diff_rad_mag[std_diff_rad_mag .== 0] .= NaN

std_diff_tot_u[std_diff_tot_u .== 0]     .= NaN
std_diff_tot_v[std_diff_tot_v .== 0]     .= NaN
std_diff_tot_mag[std_diff_tot_mag .== 0] .= NaN

std_diff_LS_u[std_diff_LS_u .== 0]       .= NaN
std_diff_LS_v[std_diff_LS_v .== 0]       .= NaN
std_diff_LS_mag[std_diff_LS_mag .== 0]   .= NaN

# ------------------------------------------------------------------------------------
# SAVE MEAN AND STANDARD DEVIATION MAPS TO NETCDF
# ------------------------------------------------------------------------------------

"""
    create_netcdf_comparison(output_file, lon_grid, lat_grid, data_rad, data_tot, data_LS; metric="diff")

Writes a NetCDF file with the 2D maps of one metric for the three
reconstruction methods compared against MEDSEA.

# Arguments
- `output_file`                : Path of the NetCDF file to create
- `lon_grid, lat_grid`         : 2D longitude and latitude grids
- `data_rad, data_tot, data_LS`: 2D maps for DIVAnd-radials, DIVAnd-totals and LS
- `metric`                     : `"diff"` (temporal mean of the magnitude difference)
                                 or `"std"` (temporal standard deviation of the
                                 magnitude difference)
"""
function create_netcdf_comparison(output_file, lon_grid, lat_grid, data_rad, data_tot, data_LS; metric="diff")
    
    # Variable names and metadata depending on the metric type
    if metric == "diff"
        var_prefix = "diff"
        var_suffix = "_mag_mean"
        long_name_prefix = "Mean speed anomaly"
        description_prefix = "Temporal mean of speed magnitude difference"
        title = "Mean velocity magnitude differences - Comparison of reconstruction methods"
    elseif metric == "std"
        var_prefix = "std_diff"
        var_suffix = "_mag"
        long_name_prefix = "Standard deviation of mean anomaly"
        description_prefix = "Standard deviation of speed magnitude difference"
        title = "Standard deviation of velocity magnitude differences - Comparison of reconstruction methods"
    else
        error("metric debe ser 'diff' o 'std'")
    end
    
    NCDataset(output_file, "c") do ds

        # Define dimensions
        defDim(ds, "lon", size(lon_grid, 1))
        defDim(ds, "lat", size(lon_grid, 2))
        
        # Save coordinate grids
        defVar(ds, "longitude", lon_grid, ("lon", "lat"), attrib = Dict(
            "long_name" => "Longitude",
            "units" => "degrees_east",
            "standard_name" => "longitude"
        ))
        
        defVar(ds, "latitude", lat_grid, ("lon", "lat"), attrib = Dict(
            "long_name" => "Latitude", 
            "units" => "degrees_north",
            "standard_name" => "latitude"
        ))
        
        # Save DIVAnd-radials variable
        defVar(ds, var_prefix * "_rad" * var_suffix, data_rad, ("lon", "lat"), attrib = Dict(
            "long_name" => "$long_name_prefix (DIVAnd-radials vs MEDSEA)",
            "units" => "m/s",
            "description" => "$description_prefix between DIVAnd reconstruction from radials and MEDSEA model"
        ))
        
        # Save DIVAnd-totals variable
        defVar(ds, var_prefix * "_tot" * var_suffix, data_tot, ("lon", "lat"), attrib = Dict(
            "long_name" => "$long_name_prefix (DIVAnd-totals vs MEDSEA)",
            "units" => "m/s",
            "description" => "$description_prefix between DIVAnd reconstruction from totals and MEDSEA model"
        ))
        
        # Save LS variable
        defVar(ds, var_prefix * "_LS" * var_suffix, data_LS, ("lon", "lat"), attrib = Dict(
            "long_name" => "$long_name_prefix (LS vs MEDSEA)",
            "units" => "m/s", 
            "description" => "$description_prefix between Least Squares reconstruction and MEDSEA model"
        ))
        
        # Global attributes
        ds.attrib["title"] = title
        ds.attrib["institution"] = "ICM-CSIC"
        ds.attrib["source"] = "HF Radar data comparison with MEDSEA model"
        ds.attrib["history"] = "Created on $(now())"
        ds.attrib["references"] = "Catalan Sea HF Radar network"
        ds.attrib["Conventions"] = "CF-1.6"
    end
    
    println("Archivo NetCDF creado: $output_file")
end

# --- Mean differences ---
create_netcdf_comparison(
    "../data/january_2026/stats_data/matrices_mean_mag.nc",
    lon_grid_LS, lat_grid_LS,
    diff_rad_mag_mean, diff_tot_mag_mean, diff_LS_mag_mean,
    metric="diff"
)

# --- Standard deviations ---
create_netcdf_comparison(
    "../data/january_2026/stats_data/matrices_std_mag.nc",
    lon_grid_LS, lat_grid_LS,
    std_diff_rad_mag, std_diff_tot_mag, std_diff_LS_mag,
    metric="std"
)