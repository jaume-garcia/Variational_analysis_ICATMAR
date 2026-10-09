using PyCall
using CSV, DataFrames
using DelimitedFiles
using NCDatasets
<<<<<<< Updated upstream
 
# -----------------------------------------------------------------------
# Script: loop_stats_10_days.jl
# Purpose: Computes domain-averaged physical oceanographic statistics
#          (divergence, vorticity, kinetic energy) and their Pearson
#          correlations with the Copernicus MEDSEA ground truth for three
#          HF radar velocity reconstruction methods (DIVAnd-radials,
#          DIVAnd-totals, and Least Squares) over a 264-step time series.
#
# Pipeline:
#   1. Load the ICATMAR bathymetry grid, LS grid, Copernicus model, and
#      DIVAnd output fields.
#   2. For each time step:
#      a. Read the LS total velocity file and filter invalid observations.
#      b. Interpolate the Copernicus field to the LS grid.
#      c. Map all velocity fields to the common LS grid using
#         find_total_points_to_LS_grid.
#      d. Compute vorticity, divergence, and kinetic energy for every method.
#      e. Compute domain-averaged values and correlations against MEDSEA.
#   3. Write all statistics to an ASCII file.
#   4. Print a summary table of temporal-mean correlation coefficients.
# -----------------------------------------------------------------------
 
# --- Julia utility modules ---
=======


# Julia scripts
>>>>>>> Stashed changes
include("../utils/reading_obs.jl")
include("../utils/divand_process.jl")
include("../utils/visualization_vel.jl")
include("stats_module.jl")
<<<<<<< Updated upstream
 
# --- Add the Python utility directory to sys.path and import modules ---
=======

# Add Python utilities directory to Python path
>>>>>>> Stashed changes
push!(PyVector(pyimport("sys")."path"), "../utils")
 
hf = pyimport("HFradar_data")
rd = pyimport("read_data")
ma = pyimport("mathematics")
 
# Create an instance of HFRadarProcessor (used for reading the LS grid)
processor = hf.HFRadarProcessor()
<<<<<<< Updated upstream
 
# ------------------------------------------------------------------------------------
# BATHYMETRY
# ------------------------------------------------------------------------------------
 
file_icatmar = "../data/bathy.nc"
mask, h, pm, pn, xi, yi = grid_icatmar(file_icatmar)
 
# ------------------------------------------------------------------------------------
# LEAST SQUARES (LS) OUTPUT GRID
# ------------------------------------------------------------------------------------
 
grid_file = "/home/jgarcia/Projects/mar_catala/radial_reconstruction/data/real_data_radar/hfradar_totals_grid_icatmar.nc"
 
lon_LS, lat_LS, lon_grid, lat_grid = processor.read_grid_from_netcdf(grid_file)
 
println(size(lon_LS))
println(size(lat_LS))
 
# Build the full 2D LS coordinate meshgrid
lat_grid_LS, lon_grid_LS = meshgrid(lat_LS, lon_LS)
 
println(size(lat_grid_LS))
 
# ------------------------------------------------------------------------------------
# COPERNICUS MEDSEA GROUND TRUTH
# ------------------------------------------------------------------------------------
 
file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"
 
# Read all time steps at once (@time measures the read time)
@time lon_cop, lat_cop, u_cop, v_cop = rd.read_nc_vel(file_cop, "lon", "lat", "u_data", "v_data", "0:")
 
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
 
# Interpolate the ICATMAR land/sea mask to the LS grid (used for masking later)
mask_interp = ma.interp_grid_eta(xi, yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])
 
=======

# ------------------------------------------------------------------------------------

# BATHYMETRY
file_icatmar = "../data/bathy.nc"

mask,h,pm,pn,xi,yi = grid_icatmar(file_icatmar);

# ------------------------------------------------------------------------------------

# LS grid

grid_file = "../data/hfradar_totals_grid_icatmar.nc"

lon_LS, lat_LS,lon_grid, lat_grid = processor.read_grid_from_netcdf(grid_file)

lat_grid_LS,lon_grid_LS = meshgrid(lat_LS,lon_LS)

println(size(lat_grid_LS))

# ------------------------------------------------------------------------------------

# COPERNICUS GROUND TRUTH

file_cop = "../data/january_2026/data_medsea/all_data_january_2026.nc"

lon_cop, lat_cop,u_cop,v_cop = rd.read_nc_vel(file_cop,"lon","lat","u_data","v_data","0:")

lon_mesh_model_cop, lat_mesh_model_cop = meshgrid(lon_cop[:,:], lat_cop[:,:])

lon_mesh_model_cop = lon_mesh_model_cop .*(180/pi)
lat_mesh_model_cop = lat_mesh_model_cop .*(180/pi)

u_model_cop = replace(u_cop, 1.0f20 => NaN32)
v_model_cop = replace(v_cop, 1.0f20 => NaN32)

println("COPERNICUS read")

# ------------------------------------------------------------------------------------

mask_interp = ma.interp_grid_eta(xi,yi, mask, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[])

>>>>>>> Stashed changes
divand_file = "../data/january_2026/divand/divand_field.nc"
 
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
<<<<<<< Updated upstream
 
uri = data.u_radial   # DIVAnd east velocity from radials
vri = data.v_radial   # DIVAnd north velocity from radials
uti = data.u_total    # DIVAnd east velocity from totals
vti = data.v_total    # DIVAnd north velocity from totals
 
# ------------------------------------------------------------------------------------
# LOCAL FUNCTION: MAP SCATTERED LS POINTS TO THE 2D LS GRID
# ------------------------------------------------------------------------------------
 
"""
    find_total_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS,
                                  u_interp_cop, v_interp_cop, u_array, v_array,
                                  uri_timestep, vri_timestep, uti_timestep, vti_timestep)
 
For each scattered LS observation point, finds the closest node on the LS grid
and assembles four co-located 2D velocity fields (Copernicus, DIVAnd-radials,
DIVAnd-totals, and LS). Also extracts 1D filtered vectors for statistical
comparisons (NaN points are excluded).
 
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
function find_total_points_to_LS_grid(lon_array, lat_array, lon_grid_LS, lat_grid_LS,
                                       u_interp_cop, v_interp_cop, u_array, v_array,
                                       uri_timestep, vri_timestep, uti_timestep, vti_timestep)
 
    tolerance = 1e-6
 
    # 1D vectors to store co-located values at each LS observation point
    u_interp_cop_at_points = zeros(length(lon_array))
    v_interp_cop_at_points = zeros(length(lon_array))
    uri_at_points          = zeros(length(lon_array))
    vri_at_points          = zeros(length(lon_array))
    uti_at_points          = zeros(length(lon_array))
    vti_at_points          = zeros(length(lon_array))
    lon_array_at_points    = zeros(length(lon_array))
    lat_array_at_points    = zeros(length(lon_array))
 
    # 2D arrays for the co-located velocity fields (NaN-initialised)
    u_rad_2d   = fill(NaN, size(lon_grid_LS))
    v_rad_2d   = fill(NaN, size(lon_grid_LS))
 
    u_tot_2d   = fill(NaN, size(lon_grid_LS))
    v_tot_2d   = fill(NaN, size(lon_grid_LS))
 
    u_LS_2d    = fill(NaN, size(lon_grid_LS))
    v_LS_2d    = fill(NaN, size(lon_grid_LS))
 
    u_model_2d = fill(NaN, size(lon_grid_LS))
    v_model_2d = fill(NaN, size(lon_grid_LS))
 
    # Counters for diagnostic output
    points_found     = 0
    points_not_found = 0
 
    for j = 1:length(lon_array)
 
        # Find the closest LS grid node using Euclidean distance in degree space
        distances    = sqrt.((lon_grid_LS .- lon_array[j]).^2 .+ (lat_grid_LS .- lat_array[j]).^2)
        min_distance = minimum(distances)
        idx_closest  = argmin(distances)
        idx_lon, idx_lat = Tuple(idx_closest)
 
        # Verify whether the closest node coincides with the observation point
        lon_closest   = lon_grid_LS[idx_lon, idx_lat]
        lat_closest   = lat_grid_LS[idx_lon, idx_lat]
        is_grid_point = (abs(lon_closest - lon_array[j]) < tolerance) &&
                        (abs(lat_closest - lat_array[j]) < tolerance)
 
=======

uri = data.u_radial
vri = data.v_radial
uti = data.u_total
vti = data.v_total

# ------------------------------------------------------------------------------------


function find_total_points_to_LS_grid(lon_array,lat_array,lon_grid_LS,lat_grid_LS,u_interp_cop,v_interp_cop,u_array,v_array,uri_timestep,vri_timestep,uti_timestep,vti_timestep)

    tolerance = 1e-6
    
    # Vectores 1D para almacenar valores en los puntos de observación
    u_interp_cop_at_points = zeros(length(lon_array))
    v_interp_cop_at_points = zeros(length(lon_array))
    uri_at_points = zeros(length(lon_array))
    vri_at_points = zeros(length(lon_array))
    uti_at_points = zeros(length(lon_array))
    vti_at_points = zeros(length(lon_array))
    lon_array_at_points = zeros(length(lon_array))
    lat_array_at_points = zeros(length(lon_array))
    
    u_rad_2d = fill(NaN, size(lon_grid_LS))
    v_rad_2d = fill(NaN, size(lon_grid_LS))
    
    u_tot_2d = fill(NaN, size(lon_grid_LS))
    v_tot_2d = fill(NaN, size(lon_grid_LS))
    
    u_LS_2d = fill(NaN, size(lon_grid_LS))
    v_LS_2d = fill(NaN, size(lon_grid_LS))
    
    u_model_2d = fill(NaN, size(lon_grid_LS))
    v_model_2d = fill(NaN, size(lon_grid_LS))

    # Contador de puntos encontrados
    points_found = 0
    points_not_found = 0
    
    for j = 1:length(lon_array)
        
        # Encontrar el índice más cercano en la malla
        distances = sqrt.((lon_grid_LS .- lon_array[j]).^2 .+ (lat_grid_LS .- lat_array[j]).^2)
        min_distance = minimum(distances)
        idx_closest = argmin(distances)
        idx_lon, idx_lat = Tuple(idx_closest)
        
        # Verificar si el punto está realmente en la malla
        lon_closest = lon_grid_LS[idx_lon, idx_lat]
        lat_closest = lat_grid_LS[idx_lon, idx_lat]
        is_grid_point = (abs(lon_closest - lon_array[j]) < tolerance) && 
                        (abs(lat_closest - lat_array[j]) < tolerance)
    
>>>>>>> Stashed changes
        if is_grid_point
            points_found += 1
        else
            points_not_found += 1
<<<<<<< Updated upstream
 
            if points_not_found <= 5   # Show only the first 5 warnings
                println("WARNING: Point $j is not exactly on the grid")
                println("  Distance: $min_distance")
                println("  Target:   ($(lon_array[j]), $(lat_array[j]))")
                println("  Closest:  ($lon_closest, $lat_closest)")
            end
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
 
        # Extract co-located values into 1D vectors for statistical comparison
        u_interp_cop_at_points[j] = u_interp_cop[idx_lon, idx_lat]
        v_interp_cop_at_points[j] = v_interp_cop[idx_lon, idx_lat]
        uri_at_points[j]          = uri_timestep[idx_lon, idx_lat]
        vri_at_points[j]          = vri_timestep[idx_lon, idx_lat]
        uti_at_points[j]          = uti_timestep[idx_lon, idx_lat]
        vti_at_points[j]          = vti_timestep[idx_lon, idx_lat]
        lon_array_at_points[j]    = lon_closest
        lat_array_at_points[j]    = lat_closest
 
    end
 
    # Build a mask for points that are valid (non-NaN) in all methods simultaneously
    valid_mask = .!(isnan.(u_interp_cop_at_points) .| isnan.(v_interp_cop_at_points) .|
                    isnan.(uri_at_points)           .| isnan.(vri_at_points)           .|
                    isnan.(uti_at_points)           .| isnan.(vti_at_points)           .|
                    isnan.(u_array)                 .| isnan.(v_array))
 
    # Extract only the commonly valid co-located points for comparisons
    u_interp_cop_filtered = u_interp_cop_at_points[valid_mask]
    v_interp_cop_filtered = v_interp_cop_at_points[valid_mask]
    uri_filtered          = uri_at_points[valid_mask]
    vri_filtered          = vri_at_points[valid_mask]
    uti_filtered          = uti_at_points[valid_mask]
    vti_filtered          = vti_at_points[valid_mask]
    u_array_filtered      = u_array[valid_mask]
    v_array_filtered      = v_array[valid_mask]
    lon_array_filtered    = lon_array_at_points[valid_mask]
    lat_array_filtered    = lat_array_at_points[valid_mask]
 
    return u_model_2d, v_model_2d, u_rad_2d, v_rad_2d, u_tot_2d, v_tot_2d, u_LS_2d, v_LS_2d,
           lon_array_filtered, lat_array_filtered,
           u_interp_cop_filtered, v_interp_cop_filtered,
           uri_filtered, vri_filtered,
           uti_filtered, vti_filtered,
           u_array_filtered, v_array_filtered
end
 
# ------------------------------------------------------------------------------------
# MAIN LOOP: COMPUTE STATISTICS FOR EACH HOURLY SNAPSHOT
# ------------------------------------------------------------------------------------
 
n_times   = 264
grid_size = size(uri[1, :, :])
 
# Output matrix: one row per time step, 22 columns:
#  1  = time index
#  2–5  = domain-mean divergence (MEDSEA, radials, totals, LS)
#  6–9  = domain-mean vorticity  (MEDSEA, radials, totals, LS)
#  10–13 = domain-mean kinetic energy (MEDSEA, radials, totals, LS)
#  14–16 = divergence Pearson correlation (radials, totals, LS) vs MEDSEA
#  17–19 = vorticity Pearson correlation (radials, totals, LS) vs MEDSEA
#  20–22 = kinetic energy Pearson correlation (radials, totals, LS) vs MEDSEA
stats_rad = zeros(n_times, 22)
 
for i = 1:n_times
 
    println("ITERATION ... ", i)
 
    # ── Copernicus snapshot at time step i ────────────────────────────
    u_cop_timestep = u_model_cop[i, :, :]
    v_cop_timestep = v_model_cop[i, :, :]
 
    println("SIZE COP = ", size(u_cop_timestep))
 
    # ── DIVAnd velocity snapshots at time step i ─────────────────────
    uri_timestep = uri[i, :, :]   # Radials – east component
    vri_timestep = vri[i, :, :]   # Radials – north component
    uti_timestep = uti[i, :, :]   # Totals  – east component
    vti_timestep = vti[i, :, :]   # Totals  – north component
 
    println("SIZE URI = ", size(uri_timestep))
    println("SIZE UTI = ", size(uti_timestep))
 
    # ── Least Squares total velocities for snapshot i ────────────────
    file_tuv = ("../data/january_2026/"
                * "totals_10_days/medsea_totals_" * lpad(string(i - 1), 3, '0') * "_all_grid.txt")
 
    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)
 
    lon_array     = df.longitude
    lat_array     = df.latitude
    u_array       = df.u_total      # East velocity (m/s)
    v_array       = df.v_total      # North velocity (m/s)
    mod_vel_array = df.speed       # Velocity magnitude (m/s)
    angle_array   = df.direction       # Current direction (degrees)
    gdop_array    = df.gdop         # Geometric Dilution of Precision
 
    # Filter out geometrically poor (GDOP > 2) and physically unrealistic (|v| > 1.2 m/s) points
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)
 
    lon_array     = lon_array[valid_index]
    lat_array     = lat_array[valid_index]
    u_array       = u_array[valid_index]
    v_array       = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array   = angle_array[valid_index]
 
    println("LENGTH TOTAL VELOCITIES L3 WITH LS (NO RESTRICTIONS) = ... ", length(u_array))
    println("TOTALS COP read")
 
    # ── Interpolate Copernicus to the LS grid ────────────────────────
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(
        lon_mesh_model_cop, lat_mesh_model_cop,
        u_cop_timestep, v_cop_timestep,
        lon_grid_LS, lat_grid_LS,
        mask_coarse_grid=[], points_mode=false
    )
 
    println("\n=== VORTICITY AND DIVERGENCE ANALYSIS ===")
 
    # ── Map all fields to the common LS grid ─────────────────────────
    u_model_2d, v_model_2d, u_rad_2d, v_rad_2d, u_tot_2d, v_tot_2d, u_LS_2d, v_LS_2d,
    lon_array_filtered, lat_array_filtered,
    u_interp_cop_filtered, v_interp_cop_filtered,
    uri_filtered, vri_filtered,
    uti_filtered, vti_filtered,
    u_array_filtered, v_array_filtered = find_total_points_to_LS_grid(
        lon_array, lat_array, lon_grid_LS, lat_grid_LS,
        u_interp_cop, v_interp_cop, u_array, v_array,
        uri_timestep, vri_timestep, uti_timestep, vti_timestep
    )
 
    # ── Vorticity (∂v/∂x − ∂u/∂y) for each method ───────────────────
    vorticity_1d_cop = calculate_vorticity(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    vorticity_1d_rad = calculate_vorticity(u_rad_2d,   v_rad_2d,   lon_grid_LS, lat_grid_LS)
    vorticity_1d_tot = calculate_vorticity(u_tot_2d,   v_tot_2d,   lon_grid_LS, lat_grid_LS)
    vorticity_1d_LS  = calculate_vorticity(u_LS_2d,    v_LS_2d,    lon_grid_LS, lat_grid_LS)
 
    # ── Divergence (∂u/∂x + ∂v/∂y) for each method ──────────────────
    divergence_1d_cop = calculate_divergence(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    divergence_1d_rad = calculate_divergence(u_rad_2d,   v_rad_2d,   lon_grid_LS, lat_grid_LS)
    divergence_1d_tot = calculate_divergence(u_tot_2d,   v_tot_2d,   lon_grid_LS, lat_grid_LS)
    divergence_1d_LS  = calculate_divergence(u_LS_2d,    v_LS_2d,    lon_grid_LS, lat_grid_LS)
 
    # Masks for grid points valid (non-NaN) in all four methods simultaneously
    valid_vort_mask = .!isnan.(vorticity_1d_cop)  .& .!isnan.(vorticity_1d_rad)  .&
                      .!isnan.(vorticity_1d_tot)  .& .!isnan.(vorticity_1d_LS)
    valid_div_mask  = .!isnan.(divergence_1d_cop) .& .!isnan.(divergence_1d_rad) .&
                      .!isnan.(divergence_1d_tot) .& .!isnan.(divergence_1d_LS)
 
    # Extract clean arrays for statistics (NaN-free)
    vorticity_model_clean = vorticity_1d_cop[valid_vort_mask]
    vorticity_rad_clean   = vorticity_1d_rad[valid_vort_mask]
    vorticity_tot_clean   = vorticity_1d_tot[valid_vort_mask]
    vorticity_LS_clean    = vorticity_1d_LS[valid_vort_mask]
 
    divergence_model_clean = divergence_1d_cop[valid_div_mask]
    divergence_rad_clean   = divergence_1d_rad[valid_div_mask]
    divergence_tot_clean   = divergence_1d_tot[valid_div_mask]
    divergence_LS_clean    = divergence_1d_LS[valid_div_mask]
 
    # ── Domain-averaged divergence ────────────────────────────────────
    div_model_mean = mean(divergence_model_clean)
    div_rad_mean   = mean(divergence_rad_clean)
    div_tot_mean   = mean(divergence_tot_clean)
    div_LS_mean    = mean(divergence_LS_clean)
 
    # ── Domain-averaged vorticity ─────────────────────────────────────
    vort_model_mean = mean(vorticity_model_clean)
    vort_rad_mean   = mean(vorticity_rad_clean)
    vort_tot_mean   = mean(vorticity_tot_clean)
    vort_LS_mean    = mean(vorticity_LS_clean)
 
    println("AVERAGES:")
    println("DIV COP = ... ", div_model_mean, " s⁻¹")
    println("DIV RAD = ... ", div_rad_mean,   " s⁻¹")
    println("DIV TOT = ... ", div_tot_mean,   " s⁻¹")
    println("DIV LS  = ... ", div_LS_mean,    " s⁻¹")
    println(" ")
    println("VORT COP = ... ", vort_model_mean, " s⁻¹")
    println("VORT RAD = ... ", vort_rad_mean,   " s⁻¹")
    println("VORT TOT = ... ", vort_tot_mean,   " s⁻¹")
    println("VORT LS  = ... ", vort_LS_mean,    " s⁻¹")
 
    # ── Pearson correlations (method vs MEDSEA) ──────────────────────
    div_corr_rad = cor(divergence_rad_clean, divergence_model_clean)
    div_corr_tot = cor(divergence_tot_clean, divergence_model_clean)
    div_corr_LS  = cor(divergence_LS_clean,  divergence_model_clean)
 
    vort_corr_rad = cor(vorticity_rad_clean, vorticity_model_clean)
    vort_corr_tot = cor(vorticity_tot_clean, vorticity_model_clean)
    vort_corr_LS  = cor(vorticity_LS_clean,  vorticity_model_clean)
 
    # ── Kinetic energy (KE = 0.5*(u² + v²)) for each method ─────────
    ke_model = calculate_kinetic_energy(u_model_2d, v_model_2d)
    ke_rad   = calculate_kinetic_energy(u_rad_2d,   v_rad_2d)
    ke_tot   = calculate_kinetic_energy(u_tot_2d,   v_tot_2d)
    ke_LS    = calculate_kinetic_energy(u_LS_2d,    v_LS_2d)
 
    println("KE calculated")
 
    # Separate validity masks for each method's KE field
    valid_mask_cop = .!(isnan.(ke_model))
    valid_mask_rad = .!(isnan.(ke_rad))
    valid_mask_tot = .!(isnan.(ke_tot))
    valid_mask_LS  = .!(isnan.(ke_tot))   # Note: uses ke_tot mask (matches original)
 
    ke_model_clean = ke_model[valid_mask_cop]
    ke_rad_clean   = ke_rad[valid_mask_rad]
    ke_tot_clean   = ke_tot[valid_mask_tot]
    ke_LS_clean    = ke_LS[valid_mask_tot]
 
    # Domain-averaged kinetic energy
    ke_model_mean = mean(ke_model_clean)
    ke_rad_mean   = mean(ke_rad_clean)
    ke_tot_mean   = mean(ke_tot_clean)
    ke_LS_mean    = mean(ke_LS_clean)
 
    println("KE COP = ... ", ke_model_mean, " m²/s²")
    println("KE RAD = ... ", ke_rad_mean,   " m²/s²")
    println("KE TOT = ... ", ke_tot_mean,   " m²/s²")
    println("KE LS  = ... ", ke_LS_mean,    " m²/s²")
 
    # Combined mask for KE correlation (all methods must be valid simultaneously)
    valid_mask_ke = valid_mask_cop .& valid_mask_rad .& valid_mask_tot .& valid_mask_LS
 
    ke_model_clean_all = ke_model[valid_mask_ke]
    ke_rad_clean_all   = ke_rad[valid_mask_ke]
    ke_tot_clean_all   = ke_tot[valid_mask_ke]
    ke_LS_clean_all    = ke_LS[valid_mask_ke]
 
    # Pearson correlation of KE with MEDSEA
    ke_corr_rad = cor(ke_rad_clean_all, ke_model_clean_all)
    ke_corr_tot = cor(ke_tot_clean_all, ke_model_clean_all)
    ke_corr_LS  = cor(ke_LS_clean_all,  ke_model_clean_all)
 
    println("KE CORR RAD = ... ", ke_corr_rad)
    println("KE CORR TOT = ... ", ke_corr_tot)
    println("KE CORR LS  = ... ", ke_corr_LS)
 
    # ── Store statistics for this time step ──────────────────────────
    stats_rad[i, :] = [
        i - 1,
        div_model_mean,  div_rad_mean,  div_tot_mean,  div_LS_mean,
        vort_model_mean, vort_rad_mean, vort_tot_mean, vort_LS_mean,
        ke_model_mean,   ke_rad_mean,   ke_tot_mean,   ke_LS_mean,
        div_corr_rad,    div_corr_tot,  div_corr_LS,
        vort_corr_rad,   vort_corr_tot, vort_corr_LS,
        ke_corr_rad,     ke_corr_tot,   ke_corr_LS
    ]
 
end
 
# ------------------------------------------------------------------------------------
# SAVE STATISTICS TO ASCII FILE
# ------------------------------------------------------------------------------------
 
output_file = "../data/january_2026/stats_data/all_stats_phys_grid_icatmar.txt"
writedlm(output_file, stats_rad, ' ')
 
println("ASCII file generated: ", output_file)
 
# ------------------------------------------------------------------------------------
# TEMPORAL MEAN CORRELATION COEFFICIENTS — SUMMARY TABLE
#
# Column layout of stats_rad:
#   1 = time index
#   2–5   = divergence (MEDSEA, radials, totals, LS)
#   6–9   = vorticity  (MEDSEA, radials, totals, LS)
#   10–13 = kinetic energy (MEDSEA, radials, totals, LS)
#   14–16 = divergence correlation (radials, totals, LS vs MEDSEA)
#   17–19 = vorticity  correlation (radials, totals, LS vs MEDSEA)
#   20–22 = kinetic energy correlation (radials, totals, LS vs MEDSEA)
# ------------------------------------------------------------------------------------
 
using Statistics
using Printf
 
# Compute temporal means of the Pearson correlation columns
mean_div_corr_LS   = mean(stats_rad[:, 16])
mean_div_corr_rad  = mean(stats_rad[:, 14])
mean_div_corr_tot  = mean(stats_rad[:, 15])
 
mean_vort_corr_LS  = mean(stats_rad[:, 19])
mean_vort_corr_rad = mean(stats_rad[:, 17])
mean_vort_corr_tot = mean(stats_rad[:, 18])
 
mean_ke_corr_LS    = mean(stats_rad[:, 22])
mean_ke_corr_rad   = mean(stats_rad[:, 20])
mean_ke_corr_tot   = mean(stats_rad[:, 21])
 
# Print the summary table
println("\n", "="^65)
println("TEMPORAL MEAN CORRELATION COEFFICIENTS")
println("="^65)
println(@sprintf("%-25s  %-10s  %-10s  %-10s", "Variable", "LS", "Radials", "Totals"))
println("-"^65)
println(@sprintf("%-25s  %-10.4f  %-10.4f  %-10.4f", "Divergence",     mean_div_corr_LS,  mean_div_corr_rad,  mean_div_corr_tot))
println(@sprintf("%-25s  %-10.4f  %-10.4f  %-10.4f", "Vorticity",      mean_vort_corr_LS, mean_vort_corr_rad, mean_vort_corr_tot))
println(@sprintf("%-25s  %-10.4f  %-10.4f  %-10.4f", "Kinetic energy", mean_ke_corr_LS,   mean_ke_corr_rad,   mean_ke_corr_tot))
println("="^65)
=======
            
            if points_not_found <= 5  # Mostrar solo los primeros 5 avisos
                println("WARNING: Punto $j no está exactamente en la malla")
                println("  Distancia: $min_distance")
                println("  Objetivo: ($(lon_array[j]), $(lat_array[j]))")
                println("  Más cercano: ($lon_closest, $lat_closest)")
            end
        end
        
        u_model_2d[idx_lon, idx_lat] = u_interp_cop[idx_lon, idx_lat]
        v_model_2d[idx_lon, idx_lat] = v_interp_cop[idx_lon, idx_lat] 
        
        u_rad_2d[idx_lon, idx_lat] = uri_timestep[idx_lon, idx_lat]
        v_rad_2d[idx_lon, idx_lat] = vri_timestep[idx_lon, idx_lat]
        
        u_tot_2d[idx_lon, idx_lat] = uti_timestep[idx_lon, idx_lat]
        v_tot_2d[idx_lon, idx_lat] = vti_timestep[idx_lon, idx_lat]
        
        u_LS_2d[idx_lon, idx_lat] = u_array[j]
        v_LS_2d[idx_lon, idx_lat] = v_array[j]
        
        # Extraer valores en el punto más cercano
        u_interp_cop_at_points[j] = u_interp_cop[idx_lon, idx_lat]
        v_interp_cop_at_points[j] = v_interp_cop[idx_lon, idx_lat]
        uri_at_points[j] = uri_timestep[idx_lon, idx_lat]
        vri_at_points[j] = vri_timestep[idx_lon, idx_lat]
        uti_at_points[j] = uti_timestep[idx_lon, idx_lat]
        vti_at_points[j] = vti_timestep[idx_lon, idx_lat]
        lon_array_at_points[j] = lon_closest
        lat_array_at_points[j] = lat_closest
        
    end
    
    # Filtrar puntos válidos (sin NaN)
    valid_mask = .!(isnan.(u_interp_cop_at_points) .| isnan.(v_interp_cop_at_points) .|
                isnan.(uri_at_points) .| isnan.(vri_at_points) .|
                isnan.(uti_at_points) .| isnan.(vti_at_points) .|
                isnan.(u_array) .| isnan.(v_array))
    
    # Vectores filtrados para cálculo
    u_interp_cop_filtered = u_interp_cop_at_points[valid_mask]
    v_interp_cop_filtered = v_interp_cop_at_points[valid_mask]
    uri_filtered = uri_at_points[valid_mask]
    vri_filtered = vri_at_points[valid_mask]
    uti_filtered = uti_at_points[valid_mask]
    vti_filtered = vti_at_points[valid_mask]
    u_array_filtered = u_array[valid_mask]
    v_array_filtered = v_array[valid_mask]
    lon_array_filtered = lon_array_at_points[valid_mask]
    lat_array_filtered = lat_array_at_points[valid_mask]
    
    return u_model_2d,v_model_2d,u_rad_2d,v_rad_2d,u_tot_2d,v_tot_2d,u_LS_2d,v_LS_2d,
    
    lon_array_filtered,lat_array_filtered,u_interp_cop_filtered,v_interp_cop_filtered,uri_filtered,vri_filtered,uti_filtered,vti_filtered,u_array_filtered,v_array_filtered
    
end




# ------------------------------------------------------------------------------------


n_times = 264
grid_size = size(uri[1, :, :])

stats_rad = zeros(n_times, 22) # time, div cop, div rad, div tot, vort cop, vort rad, vort tot, kin cop, kin rad, kin tot, correl div rad, correl div tot, correl vort rad, correl vort tot, correl kin rad, correl kin tot

# Acumuladores de vorticidad (modelo, divand-radials, divand-totals, LS) para todos los timesteps
vorticity_model_all = fill(NaN, n_times, grid_size...)
vorticity_rad_all   = fill(NaN, n_times, grid_size...)
vorticity_tot_all   = fill(NaN, n_times, grid_size...)
vorticity_LS_all    = fill(NaN, n_times, grid_size...)

u_model_all = fill(NaN, n_times, grid_size...)
v_model_all = fill(NaN, n_times, grid_size...)
u_rad_all   = fill(NaN, n_times, grid_size...)
v_rad_all   = fill(NaN, n_times, grid_size...)
u_tot_all   = fill(NaN, n_times, grid_size...)
v_tot_all   = fill(NaN, n_times, grid_size...)
u_LS_all    = fill(NaN, n_times, grid_size...)
v_LS_all    = fill(NaN, n_times, grid_size...)

for i=1:n_times

    println("ITERATION ... ",i)

    # COPERNICUS SNAPSHOT VELOCITY
    u_cop_timestep = u_model_cop[i,:,:]
    v_cop_timestep = v_model_cop[i,:,:]
    
    println("SIZE COP = ",size(u_cop_timestep))
    
    # DIVAND VELOCITIES (RADIAL AND TOTAL)
    uri_timestep = uri[i,:,:]
    vri_timestep = vri[i,:,:]
    uti_timestep = uti[i,:,:]
    vti_timestep = vti[i,:,:]
    
    println("SIZE URI = ",size(uri_timestep))
    println("SIZE UTI = ",size(uti_timestep))
    
    # TOTAL VELOCITIES (L3)
    file_tuv = "../data/january_2026/totals_10_days/medsea_totals_"*lpad(string(i-1), 3, '0')*"_all_grid.txt"

    df = CSV.read(file_tuv, DataFrame; delim='\t', comment="#", ignorerepeated=true)

    lon_array = df.longitude
    lat_array = df.latitude
    u_array = (df.u_total) 
    v_array = (df.v_total) 
    mod_vel_array = (df.speed) 
    angle_array = df.direction
    gdop_array = df.gdop
    
    valid_index = (gdop_array .<= 2.0) .& (mod_vel_array .<= 1.2)

    lon_array = lon_array[valid_index]
    lat_array = lat_array[valid_index]
    u_array = u_array[valid_index]
    v_array = v_array[valid_index]
    mod_vel_array = mod_vel_array[valid_index]
    angle_array = angle_array[valid_index]

    println("LENGTH TOTAL VELOCITIES L3 WITH JAUME LS (NO RESTRICTIONS) = ... ", length(u_array));
    
    println("TOTALS COP read");

    # INTERPOLATION (Copernicus ground truth to LS grid)
    
    u_interp_cop, v_interp_cop = ma.interp_grid_vel(lon_mesh_model_cop, lat_mesh_model_cop, u_cop_timestep, v_cop_timestep, lon_grid_LS, lat_grid_LS, mask_coarse_grid=[], points_mode=false)

 
    println("\n=== VORTICITY AND DIVERGENCE ANALYSIS ===")
    
    u_model_2d,v_model_2d,u_rad_2d,v_rad_2d,u_tot_2d,v_tot_2d,u_LS_2d,v_LS_2d,lon_array_filtered,lat_array_filtered,u_interp_cop_filtered,v_interp_cop_filtered,uri_filtered,vri_filtered,uti_filtered,vti_filtered,
    u_array_filtered,v_array_filtered  = find_total_points_to_LS_grid(lon_array,lat_array,lon_grid_LS,lat_grid_LS, u_interp_cop,v_interp_cop,u_array,v_array,uri_timestep,vri_timestep,uti_timestep,vti_timestep)
    
    
    u_model_all[i, :, :] = u_model_2d
    v_model_all[i, :, :] = v_model_2d
    u_rad_all[i, :, :]   = u_rad_2d
    v_rad_all[i, :, :]   = v_rad_2d
    u_tot_all[i, :, :]   = u_tot_2d
    v_tot_all[i, :, :]   = v_tot_2d
    u_LS_all[i, :, :]    = u_LS_2d
    v_LS_all[i, :, :]    = v_LS_2d
    
    
    vorticity_1d_cop = calculate_vorticity(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    vorticity_1d_rad = calculate_vorticity(u_rad_2d,v_rad_2d, lon_grid_LS, lat_grid_LS)
    vorticity_1d_tot = calculate_vorticity(u_tot_2d,v_tot_2d, lon_grid_LS, lat_grid_LS)
    vorticity_1d_LS = calculate_vorticity(u_LS_2d,v_LS_2d, lon_grid_LS, lat_grid_LS)

    # Guardar las vorticidades de este timestep en los acumuladores
    vorticity_model_all[i, :, :] = vorticity_1d_cop
    vorticity_rad_all[i, :, :]   = vorticity_1d_rad
    vorticity_tot_all[i, :, :]   = vorticity_1d_tot
    vorticity_LS_all[i, :, :]    = vorticity_1d_LS

    
    divergence_1d_cop = calculate_divergence(u_model_2d, v_model_2d, lon_grid_LS, lat_grid_LS)
    divergence_1d_rad = calculate_divergence(u_rad_2d,v_rad_2d, lon_grid_LS, lat_grid_LS)
    divergence_1d_tot = calculate_divergence(u_tot_2d,v_tot_2d, lon_grid_LS, lat_grid_LS)
    divergence_1d_LS = calculate_divergence(u_LS_2d,v_LS_2d, lon_grid_LS, lat_grid_LS)

    # Valid masks for vorticity and divergence (excluding NaN values)
    valid_vort_mask = .!isnan.(vorticity_1d_cop) .& .!isnan.(vorticity_1d_rad) .& .!isnan.(vorticity_1d_tot) .& .!isnan.(vorticity_1d_LS)
    valid_div_mask = .!isnan.(divergence_1d_cop) .& .!isnan.(divergence_1d_rad) .& .!isnan.(divergence_1d_tot) .& .!isnan.(divergence_1d_LS)

    # Extract clean arrays for statistics
    
    vorticity_model_clean = vorticity_1d_cop[valid_vort_mask]
    vorticity_rad_clean = vorticity_1d_rad[valid_vort_mask]
    vorticity_tot_clean = vorticity_1d_tot[valid_vort_mask]
    vorticity_LS_clean = vorticity_1d_LS[valid_vort_mask]
    
    divergence_model_clean = divergence_1d_cop[valid_div_mask]
    divergence_rad_clean = divergence_1d_rad[valid_div_mask]
    divergence_tot_clean = divergence_1d_tot[valid_div_mask]
    divergence_LS_clean = divergence_1d_LS[valid_div_mask]
    
    div_model_mean = mean(divergence_model_clean)
    div_rad_mean = mean(divergence_rad_clean)
    div_tot_mean = mean(divergence_tot_clean)
    div_LS_mean = mean(divergence_LS_clean)
    
    vort_model_mean = mean(vorticity_model_clean)
    vort_rad_mean = mean(vorticity_rad_clean)
    vort_tot_mean = mean(vorticity_tot_clean)
    vort_LS_mean = mean(vorticity_LS_clean)
    
    println("AVERAGES:")
    println("DIV COP = ... ",div_model_mean, " s⁻¹")
    println("DIV RAD = ... ",div_rad_mean, " s⁻¹")
    println("DIV TOT = ... ",div_tot_mean, " s⁻¹")
    println("DIV LS = ... ",div_LS_mean, " s⁻¹")
    println(" ")
    println("VORT COP = ... ",vort_model_mean, " s⁻¹")
    println("VORT RAD = ... ",vort_rad_mean, " s⁻¹")
    println("VORT TOT = ... ",vort_tot_mean, " s⁻¹")
    println("VORT LS = ... ",vort_LS_mean, " s⁻¹")

    # Correlations
    div_corr_rad = cor(divergence_rad_clean, divergence_model_clean)
    div_corr_tot = cor(divergence_tot_clean, divergence_model_clean)
    div_corr_LS = cor(divergence_LS_clean,divergence_model_clean)
    
    vort_corr_rad = cor(vorticity_rad_clean, vorticity_model_clean)
    vort_corr_tot = cor(vorticity_tot_clean, vorticity_model_clean)
    vort_corr_LS = cor(vorticity_LS_clean,vorticity_model_clean)
    
    # Kinetic energy
    
    ke_model = calculate_kinetic_energy(u_model_2d, v_model_2d)
    ke_rad = calculate_kinetic_energy(u_rad_2d,v_rad_2d)
    ke_tot = calculate_kinetic_energy(u_tot_2d,v_tot_2d)
    ke_LS = calculate_kinetic_energy(u_LS_2d,v_LS_2d)
    
    println("Ek CALCULADA")
    
    # Filtrar puntos válidos (sin NaN)
    valid_mask_cop = .!(isnan.(ke_model))
    valid_mask_div_rad =  .!(isnan.(ke_rad))
    valid_mask_div_tot =  .!(isnan.(ke_tot))
    valid_mask_LS =  .!(isnan.(ke_tot))
    
    ke_model_clean = ke_model[valid_mask_cop]
    ke_rad_clean = ke_rad[valid_mask_div_rad]
    ke_tot_clean = ke_tot[valid_mask_div_tot]
    ke_LS_clean = ke_LS[valid_mask_div_tot]
    
    ke_model_mean = mean(ke_model_clean)
    ke_rad_mean = mean(ke_rad_clean)
    ke_tot_mean = mean(ke_tot_clean)
    ke_LS_mean = mean(ke_LS_clean)
    
    println("KE COP = ... ",ke_model_mean, " m²/s²")
    println("KE RAD = ... ",ke_rad_mean, " m²/s²")
    println("KE TOT = ... ",ke_tot_mean, " m²/s²")
    println("KE LS = ... ",ke_LS_mean, " m²/s²")
    
    valid_mask_ke = valid_mask_cop .& valid_mask_div_rad .& valid_mask_div_tot .& valid_mask_LS
    
    ke_model_clean_all = ke_model[valid_mask_ke]
    ke_rad_clean_all = ke_rad[valid_mask_ke]
    ke_tot_clean_all = ke_tot[valid_mask_ke]
    ke_LS_clean_all = ke_LS[valid_mask_ke]
    
    ke_corr_rad = cor(ke_rad_clean_all,ke_model_clean_all)
    ke_corr_tot = cor(ke_tot_clean_all,ke_model_clean_all)
    ke_corr_LS = cor(ke_LS_clean_all,ke_model_clean_all)
    
    println("KE CORR RAD = ... ",ke_corr_rad, " m²/s²")
    println("KE CORR TOT = ... ",ke_corr_tot, " m²/s²")
    println("KE CORR LS = ... ",ke_corr_LS, " m²/s²")
    
    
    stats_rad[i, :] = [i-1,
                        div_model_mean, div_rad_mean, div_tot_mean, div_LS_mean,
                        vort_model_mean, vort_rad_mean, vort_tot_mean, vort_LS_mean,
                        ke_model_mean, ke_rad_mean, ke_tot_mean, ke_LS_mean,
                        div_corr_rad, div_corr_tot, div_corr_LS,
                        vort_corr_rad, vort_corr_tot, vort_corr_LS,
                        ke_corr_rad, ke_corr_tot, ke_corr_LS]

end


# Creating the ASCII file
output_file = "../data/january_2026/stats_data/all_stats_grid_icatmar.txt"
writedlm(output_file, stats_rad, ' ')

println("Archivo ASCII generado: ", output_file)


# ------------------------------------------------------------------------------------
# GUARDAR VORTICIDADES (MODELO, DIVAND-RADIALS, DIVAND-TOTALS, LS) EN NETCDF
# ------------------------------------------------------------------------------------

vorticity_nc_file = "../data/january_2026/stats_data/vorticity_methods.nc"

nx, ny = grid_size

isfile(vorticity_nc_file) && rm(vorticity_nc_file)

NCDataset(vorticity_nc_file, "c") do ds

    defDim(ds, "time", n_times)
    defDim(ds, "x", nx)
    defDim(ds, "y", ny)

    v_lon = defVar(ds, "longitude", Float64, ("x", "y"))
    v_lon[:, :] = lon_grid_LS

    v_lat = defVar(ds, "latitude", Float64, ("x", "y"))
    v_lat[:, :] = lat_grid_LS

    v_time = defVar(ds, "time", Float64, ("time",))
    v_time[:] = collect(0:n_times-1)
    v_time.attrib["units"] = "time step index"

    v_model = defVar(ds, "vorticity_model", Float64, ("time", "x", "y"))
    v_model[:, :, :] = vorticity_model_all
    v_model.attrib["long_name"] = "Vorticity from Copernicus model"
    v_model.attrib["units"] = "s-1"

    v_rad = defVar(ds, "vorticity_divand_radials", Float64, ("time", "x", "y"))
    v_rad[:, :, :] = vorticity_rad_all
    v_rad.attrib["long_name"] = "Vorticity from divand radial reconstruction"
    v_rad.attrib["units"] = "s-1"

    v_tot = defVar(ds, "vorticity_divand_totals", Float64, ("time", "x", "y"))
    v_tot[:, :, :] = vorticity_tot_all
    v_tot.attrib["long_name"] = "Vorticity from divand total reconstruction"
    v_tot.attrib["units"] = "s-1"

    v_LS = defVar(ds, "vorticity_LS", Float64, ("time", "x", "y"))
    v_LS[:, :, :] = vorticity_LS_all
    v_LS.attrib["long_name"] = "Vorticity from LS (Jaume) totals"
    v_LS.attrib["units"] = "s-1"

end

println("Archivo NetCDF de vorticidades generado: ", vorticity_nc_file)


# ------------------------------------------------------------------------------------
# GUARDAR VELOCIDADES PARA ESPECTRO ENERGÍA (MODELO, DIVAND-RADIALS, DIVAND-TOTALS, LS) EN NETCDF
# ------------------------------------------------------------------------------------

velocity_nc_file = "../data/january_2026/stats_data/velocity_methods.nc"
isfile(velocity_nc_file) && rm(velocity_nc_file)
 
NCDataset(velocity_nc_file, "c") do ds
    defDim(ds, "time", n_times)
    defDim(ds, "x", nx)
    defDim(ds, "y", ny)
 
    v_lon = defVar(ds, "longitude", Float64, ("x", "y")); v_lon[:, :] = lon_grid_LS
    v_lat = defVar(ds, "latitude", Float64, ("x", "y"));  v_lat[:, :] = lat_grid_LS
 
    v_time = defVar(ds, "time", Float64, ("time",))
    v_time[:] = collect(0:n_times-1)
    v_time.attrib["units"] = "time step index"
 
    for (name, arr, long_name) in [
        ("u_model",           u_model_all, "u Copernicus (model)"),
        ("v_model",           v_model_all, "v Copernicus (model)"),
        ("u_divand_radials",  u_rad_all,   "u DIVAnd radial reconstruction"),
        ("v_divand_radials",  v_rad_all,   "v DIVAnd radial reconstruction"),
        ("u_divand_totals",   u_tot_all,   "u DIVAnd total reconstruction"),
        ("v_divand_totals",   v_tot_all,   "v DIVAnd total reconstruction"),
        ("u_LS",              u_LS_all,    "u LS (Jaume) totals"),
        ("v_LS",              v_LS_all,    "v LS (Jaume) totals"),
    ]
        vvar = defVar(ds, name, Float64, ("time", "x", "y"))
        vvar[:, :, :] = arr
        vvar.attrib["long_name"] = long_name
        vvar.attrib["units"] = "m s-1"
    end
end
 
println("Archivo NetCDF de velocidades generado: ", velocity_nc_file)





# ------------------------------------------------------------------------------------
# PROMEDIOS DE LOS COEFICIENTES DE CORRELACIÓN
# Columnas de stats_rad:
#  1=time | 2=div_cop | 3=div_rad | 4=div_tot | 5=div_LS
#  6=vort_cop | 7=vort_rad | 8=vort_tot | 9=vort_LS
#  10=ke_cop | 11=ke_rad | 12=ke_tot | 13=ke_LS
#  14=div_corr_rad | 15=div_corr_tot | 16=div_corr_LS
#  17=vort_corr_rad | 18=vort_corr_tot | 19=vort_corr_LS
#  20=ke_corr_rad  | 21=ke_corr_tot  | 22=ke_corr_LS

using Statistics
using Printf

mean_div_corr_LS  = mean(stats_rad[:, 16])
mean_div_corr_rad = mean(stats_rad[:, 14])
mean_div_corr_tot = mean(stats_rad[:, 15])

mean_vort_corr_LS  = mean(stats_rad[:, 19])
mean_vort_corr_rad = mean(stats_rad[:, 17])
mean_vort_corr_tot = mean(stats_rad[:, 18])

mean_ke_corr_LS  = mean(stats_rad[:, 22])
mean_ke_corr_rad = mean(stats_rad[:, 20])
mean_ke_corr_tot = mean(stats_rad[:, 21])

println("\n", "="^65)
println("PROMEDIOS DEL COEFICIENTE DE CORRELACIÓN")
println("="^65)
println(@sprintf("%-25s  %-10s  %-10s  %-10s", "Variable", "LS", "Radials", "Totals"))
println("-"^65)
println(@sprintf("%-25s  %-10.4f  %-10.4f  %-10.4f", "Divergence",      mean_div_corr_LS,  mean_div_corr_rad,  mean_div_corr_tot))
println(@sprintf("%-25s  %-10.4f  %-10.4f  %-10.4f", "Vorticity",       mean_vort_corr_LS, mean_vort_corr_rad, mean_vort_corr_tot))
println(@sprintf("%-25s  %-10.4f  %-10.4f  %-10.4f", "Kinetic energy",  mean_ke_corr_LS,   mean_ke_corr_rad,   mean_ke_corr_tot))
>>>>>>> Stashed changes
