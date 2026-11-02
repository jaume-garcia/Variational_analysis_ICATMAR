# Variational Analysis and Gap Filling — ICATMAR HF Radar 

This repository contains a multi-language scientific pipeline (Python, Julia, Fortran) for **reconstructing and gap-filling HF radar surface current velocity fields** in the Catalan Sea (NW Mediterranean) using the **DIVAnd** (Data-Interpolating Variational Analysis in n Dimensions) algorithm. Reconstructed fields are validated against Copernicus MEDSEA model reanalysis data used as ground truth.

---

## Table of Contents

1. [Dependencies](#dependencies)
   - [Python packages](#python-packages)
   - [Julia packages](#julia-packages)
   - [Fortran requirements](#fortran-requirements)
2. [Folder setup](#folder-setup)
3. [Repository structure](#repository-structure)
   - [`data/`](#data)
   - [`src/`](#src)
   - [`src/fortran/`](#srcfortran)
   - [`src/plot_scripts/`](#srcplot_scripts)
   - [`utils/`](#utils)
4. [How to run](#how-to-run)

---

## Dependencies

### Python packages

The Python environment requires **Python ≥ 3.8**. Install all dependencies with pip or conda:

```bash
pip install numpy pandas scipy matplotlib cartopy netCDF4 xarray seaborn julia
```

| Package | Role |
|---|---|
| `numpy` | Core numerical arrays and math |
| `pandas` | Tabular data handling and CSV I/O |
| `scipy` | Interpolation, spatial filtering, statistics |
| `matplotlib` | 2-D plotting and colormaps |
| `cartopy` | Geospatial map projections and coastlines |
| `netCDF4` | Reading/writing NetCDF files |
| `xarray` | Labelled multi-dimensional arrays (NetCDF wrappers) |
| `seaborn` | Statistical visualisation |
| `julia` | Python–Julia bridge (`pyjulia`) used to call Julia routines from Python scripts |

> **Note:** `pyjulia` requires a working Julia installation (see below). On first use, run `python -c "from julia import Julia; Julia(compiled_modules=False)"` to initialise the bridge.

---

### Julia packages

The pipeline uses **Julia ≥ 1.9**. Install all packages from the Julia REPL:

```julia
using Pkg
Pkg.add([
    "DIVAnd",
    "DIVAnd_HFRadar",
    "NCDatasets",
    "CSV",
    "DataFrames",
    "Interpolations",
    "MeshGrid",
    "NaNMath",
    "NaturalEarth",
    "PyCall",
    "PyPlot",
    "Statistics",
    "Dates",
    "DelimitedFiles",
    "Printf",
    "Random",
])
```

| Package | Role |
|---|---|
| `DIVAnd` | Core variational interpolation / gap-filling algorithm |
| `DIVAnd_HFRadar` | DIVAnd extension for HF radar radial velocities |
| `NCDatasets` | NetCDF read/write in Julia |
| `CSV` / `DataFrames` | Tabular I/O |
| `Interpolations` | Grid-to-grid spatial interpolation |
| `MeshGrid` | MATLAB-style meshgrid helper |
| `NaNMath` | NaN-aware mathematical operations |
| `NaturalEarth` | Coastline/land polygons for map plotting |
| `PyCall` / `PyPlot` | Julia–Python bridge and Matplotlib access |
| `Statistics` / `Random` | Standard library statistics and random sampling |
| `Dates` / `Printf` / `DelimitedFiles` | Standard library utilities |

> **Note:** `PyCall` and `PyPlot` must point to the same Python environment used by the Python scripts. Set the `PYTHON` environment variable before building: `ENV["PYTHON"] = "/path/to/python"; Pkg.build("PyCall")`.

---

### Fortran requirements

The `src/fortran/` sub-module is an optional high-performance reimplementation of the Least Squares (LS) combiner. It requires:

| Requirement | Notes |
|---|---|
| **gfortran ≥ 9** (or ifort / pgfortran) | Fortran 90/95/2003 compiler |
| **NetCDF-Fortran** | HDF/NetCDF library with Fortran bindings |

Install NetCDF-Fortran:

```bash
# Ubuntu / Debian
sudo apt-get install libnetcdf-dev libnetcdff-dev

# Fedora / CentOS
sudo yum install netcdf-fortran-devel

# macOS (Homebrew)
brew install netcdf netcdf-fortran
```

Compile the Fortran module:

```bash
cd src/fortran
make          # standard build
make rebuild  # clean + build
make check    # syntax check only
```

---

## Folder setup

Before downloading any data or running any script, create the required directory tree. The repository does **not** include these folders (they are listed in `.gitignore` due to file size). Run the block below from the repository root:

```bash
# ── data folders ──────────────────────────────────────────────────────────────
mkdir -p data/january_2026/totals_10_days
mkdir -p data/january_2026/divand
mkdir -p data/january_2026/gaps_data
mkdir -p data/january_2026/stats_data
mkdir -p data/january_2026/radials_10_days/csv_files_radials_medsea
mkdir -p data/january_2026/radials_10_days/csv_files_real_radar
mkdir -p data/january_2026/radials_10_days/filtered_radials

# ── figure folders ────────────────────────────────────────────────────────────
mkdir -p figures/january_2026/gaps/time_series_vel_mag
mkdir -p figures/january_2026/gaps_by_size/obs_gap_large
mkdir -p figures/january_2026/gaps_by_size/obs_gap_small
mkdir -p figures/january_2026/statistics
mkdir -p figures/january_2026/physics
mkdir -p figures/january_2026/physics/vorticity
mkdir -p figures/january_2026/medsea_GT
mkdir -p figures/january_2026/medsea_radials
```

The resulting layout is:

```
Variational_analysis_ICATMAR/
├── data/
│   └── january_2026/
│       ├── data_medsea/                        # Copernicus MEDSEA reanalysis NetCDF files (downloaded)
│       ├── totals_10_days/                     # LS total velocity outputs per 10-day window
│       ├── divand/                             # DIVAnd reconstructed velocity fields
│       ├── gaps_data/                          # Gap classification and metadata files
│       ├── stats_data/                         # Statistical comparison outputs
│       └── radials_10_days/                    # Raw HF radar radial files (.ruv) + derived CSVs
│           ├── csv_files_radials_medsea/       # Synthetic radials projected from Copernicus model
│           ├── csv_files_real_radar/           # CSVs converted from raw .ruv radial files
│           └── filtered_radials/               # Cross-antenna filtered radial text files
└── figures/
    └── january_2026/
        ├── gaps/
        │   └── time_series_vel_mag/            # Time-series plots of velocity magnitude at gap points
        ├── gaps_by_size/
        │   ├── obs_gap_large/                  # Figures for large-gap validation
        │   └── obs_gap_small/                  # Figures for small-gap validation
        ├── statistics/                         # RMSE maps, bar charts and statistical summary figures
        ├── physics/                            # Divergence, vorticity and kinetic energy plots
        ├── medsea_GT/                          # Copernicus ground-truth velocity field snapshots
        └── medsea_radials/                     # Filtered radial vector maps overlaid on MEDSEA domain
```

---

## Repository structure

```
Variational_analysis_ICATMAR/
├── data/
│   └── january_2026/
│       ├── data_medsea/                    # Copernicus MEDSEA reanalysis NetCDF files
│       ├── totals_10_days/                 # LS total velocity outputs per 10-day window
│       ├── divand/                         # DIVAnd reconstructed velocity fields
│       ├── gaps_data/                      # Gap classification and metadata files
│       ├── stats_data/                     # Statistical comparison outputs
│       └── radials_10_days/               # Raw HF radar radial files (.ruv) + derived CSVs
│           ├── csv_files_radials_medsea/  # Synthetic radials from Copernicus model
│           ├── csv_files_real_radar/      # CSVs converted from raw .ruv files
│           └── filtered_radials/          # Cross-antenna filtered radial text files
├── figures/
│   └── january_2026/
│       ├── gaps/
│       │   └── time_series_vel_mag/       # Time-series plots of velocity magnitude at gap points
│       ├── gaps_by_size/
│       │   ├── obs_gap_large/             # Figures for large-gap validation
│       │   └── obs_gap_small/             # Figures for small-gap validation
│       ├── statistics/                    # RMSE maps, bar charts and statistical summaries
│       ├── physics/                       # Divergence, vorticity and kinetic energy plots
│       ├── medsea_GT/                     # Copernicus ground-truth velocity field snapshots
│       └── medsea_radials/               # Filtered radial vector maps on MEDSEA domain
├── src/
│   ├── create_medsea_radials.py
│   ├── gap_class_LS.py
│   ├── loop_avg_diff_std_all_methods.jl
│   ├── loop_gap_filling_divand.py
│   ├── loop_gap_filling_divand_by_size.py
│   ├── loop_neighbours_gaps.jl
│   ├── loop_obs_gap_vel.jl
│   ├── loop_rmse_all_methods.jl
│   ├── loop_save_divands_icatmar_grid.jl
│   ├── loop_stats_10_days.jl
│   ├── obs_point_gap.py
│   ├── stats_module.jl
│   ├── fortran/
│   └── plot_scripts/
└── utils/
    ├── HFradar_data.py
    ├── create_csv_radar.py
    ├── create_filt_radials.py
    ├── divand_process.jl
    ├── fig_vel.py
    ├── get_medsea_data.py
    ├── mathematics.py
    ├── read_data.py
    ├── reading_obs.jl
    ├── vel_eta.py
    └── visualization_vel.jl
```

---

### `data/`

The `data/` directory is **not fully versioned** in this repository due to file size. Before running any script, you must download the required datasets and place them in the correct subfolders as described below.

---

#### Copernicus MEDSEA reanalysis data → `data/january_2026/data_medsea/`

Download daily NetCDF files of surface velocity (u, v) from the **Copernicus Marine Service**:

- **Product:** Mediterranean Sea Physics Reanalysis — `MEDSEA_MULTIYEAR_PHY_006_004`
- **Dataset ID:** `med-cmcc-cur-rean-d` (daily, 2-D surface currents, `RFVL` variable)
- **Portal:** [https://data.marine.copernicus.eu/product/MEDSEA_MULTIYEAR_PHY_006_004/services](https://data.marine.copernicus.eu/product/MEDSEA_MULTIYEAR_PHY_006_004/services)

Steps:
1. Create a free account at [https://marine.copernicus.eu](https://marine.copernicus.eu) if you do not have one.
2. Go to the product page linked above and select **Subset** to filter by time range (e.g. January 2026), variables (`uo`, `vo`), and spatial extent (NW Mediterranean, roughly lon 1°–5°E, lat 40°–43.5°N).
3. Download the resulting daily NetCDF files and place them in `data/january_2026/data_medsea/`.

Alternatively, use the **Copernicus Marine Toolbox** (CLI):
```bash
pip install copernicusmarine
copernicusmarine subset \
  --dataset-id med-cmcc-cur-rean-d \
  --variable uo --variable vo \
  --start-datetime 2026-01-14T00:00:00 \
  --end-datetime 2026-01-23T23:59:59 \
  --minimum-longitude 1.0 --maximum-longitude 5.0 \
  --minimum-latitude 40.0 --maximum-latitude 43.5 \
  --output-directory data/january_2026/data_medsea/
```

Files should follow the naming pattern already present in the repository, e.g.: `20260114_2dh-CMCC--RFVL-MFSeas10-MEDATL-b20260127_an-sv11.00.nc`

---

#### ICATMAR HF radar radial data → `data/january_2026/radials_10_days/`

Download hourly radial velocity files (`.ruv` format) from the **ICATMAR Operational Oceanography Service**:

- **Portal:** [https://www.icatmar.cat/servei-ftp/](https://www.icatmar.cat/servei-ftp/)
- **Contact / data access:** info@icatmar.cat

The ICATMAR network consists of seven coastal HF radar antennas (AREN, BEGU, CREU, GNST, PBCN, TOSS, and one at the Port of Barcelona). Raw radial files are in CODAR `.ruv` format and follow the naming convention:

```
RDLm_<STATION>_<YYYY>_<MM>_<DD>_<HHMM>_l2b.ruv
```

For example: `RDLm_AREN_2026_01_14_0000_l2b.ruv`

> **Note:** Access to raw `.ruv` radial files may require contacting ICATMAR directly, as the public viewer only displays processed products. Reach out via the contact page at [https://www.icatmar.cat/en/](https://www.icatmar.cat/en/) to request data access for research purposes.

Once downloaded, place all `.ruv` files for the desired period in `data/january_2026/radials_10_days/`.

---

#### Additional grid files (provided separately)

The following NetCDF files are also required but are not publicly available for download — they must be requested from ICATMAR or generated from the processing pipeline:

- `data/bathy.nc` — Bathymetry and ICATMAR grid metrics (mask, h, pm, pn, xi, yi)
- `data/hfradar_totals_grid_icatmar.nc` — ICATMAR Least Squares output grid with antenna coverage masks

---

### `src/`

Main processing and analysis scripts. Scripts are designed to be run from within the `src/` directory so that relative paths (`../data/`, `../utils/`) resolve correctly.

| Script | Language | Purpose |
|---|---|---|
| `create_medsea_radials.py` | Python | Reads `all_data_january_2026.nc` and projects Copernicus model total velocities onto ICATMAR antenna beam directions to generate **synthetic radials** (one CSV per snapshot) in `radials_10_days/csv_files_radials_medsea/` |
| `gap_class_LS.py` | Python | Classifies spatial gaps in the Least Squares total velocity field by size and geometry |
| `obs_point_gap.py` | Python | Extracts observation points that fall inside identified gap regions |
| `loop_gap_filling_divand.py` | Python | Main gap-filling loop: runs DIVAnd reconstruction (both radial-based and total-based) for each time step and evaluates RMSE against Copernicus ground truth inside LS gap regions |
| `loop_gap_filling_divand_by_size.py` | Python | Same as above but stratifies the evaluation by gap size category |
| `loop_save_divands_icatmar_grid.jl` | Julia | Runs DIVAnd on the ICATMAR LS grid for all hourly snapshots, reconstructing radial- and total-based velocity fields with separate parameter sets and saving them to NetCDF |
| `loop_rmse_all_methods.jl` | Julia | Computes RMSE for all three methods (DIVAnd-radials, DIVAnd-totals, LS) over the full time series against Copernicus data |
| `loop_avg_diff_std_all_methods.jl` | Julia | Computes domain-averaged velocity time series, differences (u, v and speed) and relative speed differences for DIVAnd-radials, DIVAnd-totals and LS against Copernicus; saves temporal mean and standard-deviation maps to NetCDF |
| `loop_stats_10_days.jl` | Julia | Full statistical comparison (divergence, vorticity, kinetic energy, mean, correlation) between DIVAnd and LS against Copernicus over 10-day windows |
| `loop_neighbours_gaps.jl` | Julia | Analyses spatial neighbourhood structure of gap regions and their surrounding observations |
| `loop_obs_gap_vel.jl` | Julia | Extracts and stores velocity observations surrounding each identified gap for later analysis |
| `stats_module.jl` | Julia | Shared module with statistical functions (RMSE, correlation, divergence, vorticity, KE) used by other Julia scripts |

**`src/fortran/`** — Fortran 90 reimplementation of the HF radar Least Squares total velocity combiner. See [Fortran requirements](#fortran-requirements) for compilation instructions.

Key files:

| File | Purpose |
|---|---|
| `hfradar_module.f90` | Module defining data types (`SiteInfo`, `RadialData`, `TotalVelocity`, `HFRadarProcessor`) and all computation routines (Haversine distance, LS combination, NetCDF grid reader, results writer) |
| `hfradar_main.f90` | Main program: reads radial `.txt` files, calls the processor, writes total velocity output |
| `hfradar_config.nml` | Namelist configuration file for paths, processing parameters, and file ranges |
| `Makefile` | Automated build with automatic NetCDF detection via `nf-config` |
| `compare_results.py` | Python script to numerically compare Fortran and Python LS outputs |

---

### `src/plot_scripts/`

Visualisation scripts that generate publication-quality figures. Run from within `src/plot_scripts/` or provide correct relative paths.

| Script | Language | Output |
|---|---|---|
| `plot_avg_temp_all_methods.jl` | Julia | Temporal mean velocity magnitude maps comparing DIVAnd-radials, DIVAnd-totals, and LS against Copernicus over the full time series (264 steps) |
| `plot_filtered_radials_medsea.py` | Python | Map of filtered radial vectors from all ICATMAR stations overlaid on the Mediterranean domain |
| `plot_medsea_10_days.py` | Python | Velocity field snapshots for each 10-day period |
| `plot_rms_gaps.py` | Python | RMSE bar charts for all methods evaluated inside LS gap regions |
| `plot_rms_gaps_by_size.py` | Python | RMSE breakdown by gap size category |
| `plot_rms_icatmar_grid_all_methods.py` | Python | RMSE maps on the ICATMAR grid for all reconstruction methods |
| `plot_stats_10_days_icatmar_grid.py` | Python | Statistical summary plots (divergence, vorticity, KE) over 10-day periods |
| `plot_time_series_vel_points.py` | Python | Time series of u/v velocity at specific observation points |
| `plot_vel_obs_gap.py` | Python | Velocity field visualisation centred on gap regions with surrounding observations |

---

### `utils/`

Shared utility library imported by both `src/` Python scripts and Julia scripts (via `PyCall`). Do not run these files directly.

| File | Language | Contents |
|---|---|---|
| `HFradar_data.py` | Python | `HFRadarProcessor` class: reads `.ruv` radial files, reads LS NetCDF grids, computes Least Squares total velocities, provides radar station metadata (`radars_cat()`); also contains `create_all_csv_radar()`, `read_obs_csv()`, and `create_filtered_radials()` helpers used by the preprocessing scripts |
| `get_medsea_data.py` | Python | Reads the individual daily Copernicus NetCDF files from `data_medsea/`, concatenates all time steps for the chosen period, and saves the result as a single file (`all_data_january_2026.nc`) in `data/january_2026/data_medsea/` |
| `create_csv_radar.py` | Python | Reads raw `.ruv` radial files from `radials_10_days/` for each hourly snapshot and writes one CSV per snapshot to `radials_10_days/csv_files_real_radar/` |
| `create_filt_radials.py` | Python | Applies a cross-antenna distance filter (radius 6 km) to the synthetic radial CSVs in `csv_files_radials_medsea/`: only radials that have at least one observation from a different antenna within the search radius are kept. Filtered output is written to `radials_10_days/filtered_radials/` as one text file per snapshot |
| `read_data.py` | Python | Low-level NetCDF readers for velocity, bathymetry, and mask fields |
| `mathematics.py` | Python | Spatial interpolation utilities (`interp_grid_eta`, `interp_grid_vel`), Haversine distance, grid metric calculations |
| `vel_eta.py` | Python | Velocity component transformations, projection of totals onto radial directions, η (sea-level) related computations |
| `fig_vel.py` | Python | Matplotlib/Cartopy figure helpers: velocity quiver and scalar maps with coastlines and colorbars |
| `reading_obs.jl` | Julia | Julia-side readers for radial and total NetCDF files; `grid_icatmar()` function that loads bathymetry mask and grid metrics |
| `divand_process.jl` | Julia | Wrappers around `DIVAnd` and `DIVAnd_HFRadar` calls: sets up correlation lengths, signal-to-noise ratio, and runs the analysis |
| `visualization_vel.jl` | Julia | Julia plotting helpers using PyPlot: velocity magnitude maps, difference maps, scatter plots |

---

## How to run

All scripts are designed to be executed from their own directory. The general workflow follows the steps below.

### 1. Prepare the environment

```bash
# Clone the repository
git clone https://github.com/jaume-garcia/Variational_analysis_ICATMAR.git
cd Variational_analysis_ICATMAR

# Install Python dependencies
pip install numpy pandas scipy matplotlib cartopy netCDF4 xarray seaborn julia

# Install Julia packages (from Julia REPL)
# julia -e 'using Pkg; Pkg.add([...])'  # see Dependencies section above

# (Optional) Compile Fortran module
cd src/fortran && make && cd ../..
```

Make sure the following files are present before running any script (see [data section](#data) for download instructions):
- `data/bathy.nc`
- `data/hfradar_totals_grid_icatmar.nc`
- `data/january_2026/data_medsea/*.nc`
- `data/january_2026/radials_10_days/*.ruv`

### 2. Concatenate Copernicus data into a single NetCDF

```bash
cd utils
python get_medsea_data.py
```

Reads the individual daily Copernicus NetCDF files from `data/january_2026/data_medsea/` and concatenates all time steps for the study period into a single file:

```
data/january_2026/data_medsea/all_data_january_2026.nc
```

### 3. Convert ICATMAR radial files to CSV

```bash
python create_csv_radar.py
```

Reads the raw `.ruv` radial files from `data/january_2026/radials_10_days/` for each hourly snapshot and writes one CSV per snapshot to:

```
data/january_2026/radials_10_days/csv_files_real_radar/
```

### 4. Generate synthetic radials from the Copernicus model

```bash
cd ../src
python create_medsea_radials.py
```

Reads `all_data_january_2026.nc` and the real radar CSV files, then projects the Copernicus total velocities onto the ICATMAR antenna beam directions to produce synthetic radial observations (one CSV per snapshot):

```
data/january_2026/radials_10_days/csv_files_radials_medsea/
```

### 5. Filter radials by influence radius

```bash
cd ../utils
python create_filt_radials.py
```

Applies a cross-antenna distance filter (6 km radius) to the synthetic radial CSVs. Only radials that have at least one observation from a different antenna within the search radius are retained. Filtered output (one text file per snapshot) is written to:

```
data/january_2026/radials_10_days/filtered_radials/
```

### 6. Compute LS total velocities (Fortran)

```bash
cd ../src/fortran
make
./hfradar_program
```
If you encounter compilation issues or suspect leftover object/module files from a previous build, run `make clean` before `make` to ensure a fresh compilation.

This step generates the total velocity fields using the Least Squares (LS) method, required for subsequent gap analysis.

### 7. Classify gaps in the LS field

```bash
cd ../src
python gap_class_LS.py
```

### 8. Save DIVAnd outputs on the ICATMAR grid

```bash
julia loop_save_divands_icatmar_grid.jl
```

### 9. Run DIVAnd gap filling and validation

```bash
# Standard gap filling (validates RMSE inside LS gap regions)
python loop_gap_filling_divand.py

# Gap filling stratified by gap size
python loop_gap_filling_divand_by_size.py
```

### 10. Compute statistics and RMSE across all methods

```bash
julia loop_rmse_all_methods.jl
julia loop_avg_diff_std_all_methods.jl
julia loop_stats_10_days.jl
julia loop_neighbours_gaps.jl
julia loop_obs_gap_vel.jl
```

### 11. Generate figures

```bash
# Julia plots
julia plot_scripts/plot_avg_temp_all_methods.jl

# Python plots
python plot_scripts/plot_rms_gaps.py
python plot_scripts/plot_rms_gaps_by_size.py
python plot_scripts/plot_rms_icatmar_grid_all_methods.py
python plot_scripts/plot_stats_10_days_icatmar_grid.py
python plot_scripts/plot_time_series_vel_points.py
python plot_scripts/plot_vel_obs_gap.py
python plot_scripts/plot_filtered_radials_medsea.py
python plot_scripts/plot_medsea_10_days.py
```

---

## Acknowledgments

This work was developed within the Catalan Institute for Ocean Governance Research (ICATMAR) and was supported by the European Maritime, Fisheries and Aquaculture Fund (EMFAF) with the institutional support of the grant *Severo Ochoa Centre of Excellence* accreditation (CEX2024-001494-S) funded by AEI 10.13039/501100011033. The ICATMAR is a cooperative organization between the Government of Catalonia and the Spanish National Research Council through the Institute of Marine Sciences.

This README was written with the assistance of Claude (Anthropic).

---

## Contact

**Author**
Nombre · ICM/CSIC
✉ ashhsa@sdjas.com

---

## Notes

- Scripts use **relative paths** (`../data/`, `../utils/`). Always run them from within their own directory.
- The Python–Julia bridge (`pyjulia`) adds startup overhead on the first call. Mixed-language scripts (`.py` files that import Julia, or `.jl` files that import Python via `PyCall`) require both runtimes to be correctly configured and pointing to the same environment.
- Large NetCDF outputs and raw radar files are **not versioned** in the repository due to size. The `data/` folder in the repository only contains the January 2026 test dataset.
