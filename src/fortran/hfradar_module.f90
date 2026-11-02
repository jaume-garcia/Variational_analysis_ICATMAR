!===============================================================================
! Module: HFradar_data
! Description: Provides data structures and computational methods for 
!              processing HF Radar radial velocities into total vector fields.
!===============================================================================
module HFradar_data
    implicit none
    
    ! --- Physical and Mathematical Constants ---
    real(8), parameter :: PI = 3.141592653589793d0
    real(8), parameter :: EARTH_RADIUS = 6371.0d0  ! Mean Earth radius in km
    
    ! --- Performance and Timing Variables ---
    integer(8) :: t_start, t_end, count_rate
    real(8) :: elapsed
    
    ! --- Derived Type: Radar Site Metadata ---
    type :: SiteInfo
        character(len=50) :: name     ! Station identifier
        real(8) :: latitude           ! Geographic latitude (decimal degrees)
        real(8) :: longitude          ! Geographic longitude (decimal degrees)
        integer :: flag               ! Unique antenna identification code
    end type SiteInfo
    
    ! --- Derived Type: Radial Data Storage ---
    ! Stores observed radial components from individual radar sites
    type :: RadialData
        integer :: n_points
        real(8), allocatable :: lond(:)         ! Longitude of radial vector
        real(8), allocatable :: latd(:)         ! Latitude of radial vector
        real(8), allocatable :: velu(:)         ! Eastward velocity component
        real(8), allocatable :: velv(:)         ! Northward velocity component
        real(8), allocatable :: bear(:)         ! Bearing (degrees)
        real(8), allocatable :: velo(:)         ! Radial velocity magnitude
        real(8), allocatable :: dir(:)          ! Directional angle
        integer, allocatable :: antenna_flag(:) ! Originating antenna ID
        real(8), allocatable :: bear_rad(:)     ! Bearing in radians
    end type RadialData
    
    ! --- Derived Type: Total Velocity Results ---
    ! Stores synthesized total vector current results
    type :: TotalVelocity
        real(8) :: longitude    ! Target grid longitude
        real(8) :: latitude     ! Target grid latitude
        real(8) :: u_total      ! Synthesized Eastward component (u)
        real(8) :: v_total      ! Synthesized Northward component (v)
        real(8) :: speed       ! Vector magnitude (speed)
        real(8) :: direction       ! Vector direction (oceanographic convention)
        real(8) :: gdop         ! Geometric Dilution of Precision
        integer :: n_obs        ! Number of contributing radial observations
        integer :: n_sites      ! Number of contributing radar sites
    end type TotalVelocity
    
    ! --- Derived Type: Main Processor Class ---
    ! Encapsulates the radar data and associated processing methods
    type :: HFRadarProcessor
        type(RadialData) :: radials
        type(SiteInfo), allocatable :: sites(:)
        integer :: n_sites
    contains
        procedure :: parse_text_file             ! Ingests radial text data
        procedure :: read_grid_from_netcdf       ! Loads computational grid
        procedure :: calculate_distance          ! Spherical distance (Haversine)
        procedure :: least_squares_combination   ! Radial-to-total synthesis
        procedure :: write_results_txt           ! Exports results to ASCII
        procedure :: process_text_files          ! Main workflow driver
        procedure :: cleanup                     ! Memory deallocation
    end type HFRadarProcessor
    
contains
    
    !===========================================================================
    ! Subroutine: parse_text_file
    ! Description: Parses ASCII files containing radial velocity observations.
    !              Dynamically allocates memory and identifies unique sensors.
    !===========================================================================
    subroutine parse_text_file(this, filepath, success)
    class(HFRadarProcessor), intent(inout) :: this
    character(len=*), intent(in) :: filepath
    logical, intent(out) :: success
    
    integer :: unit, ios, i, n_lines
    real(8) :: lond, latd, velu, velv, bear, velo, dir
    real(8) :: antenna_flag
    integer, allocatable :: unique_flags(:), temp_flags(:)
    integer :: n_unique, j
    logical :: flag_exists
    real(8) :: sum_lat, sum_lon
    integer :: count_flag
    character(len=500) :: line_buffer
    
    success = .false.
    
    ! Open source file for read-only access
    open(newunit=unit, file=trim(filepath), status='old', action='read', iostat=ios)
    if (ios /= 0) then
        print *, 'Error: Unable to access file ', trim(filepath)
        return
    end if
    
    ! Determine total record count for memory allocation
    n_lines = 0
    do
        read(unit, '(A)', iostat=ios) line_buffer
        if (ios /= 0) exit
        line_buffer = adjustl(line_buffer)
        if (len_trim(line_buffer) > 0) then
            n_lines = n_lines + 1
        end if
    end do
    rewind(unit)
    
    print '(A,I0,A)', 'File: ', n_lines, ' vectors successfully identified'
    
    if (n_lines == 0) then
        close(unit)
        return
    end if
    
    ! Allocate memory for radial data structures
    this%radials%n_points = n_lines
    allocate(this%radials%lond(n_lines))
    allocate(this%radials%latd(n_lines))
    allocate(this%radials%velu(n_lines))
    allocate(this%radials%velv(n_lines))
    allocate(this%radials%bear(n_lines))
    allocate(this%radials%velo(n_lines))
    allocate(this%radials%dir(n_lines))
    allocate(this%radials%antenna_flag(n_lines))
    allocate(this%radials%bear_rad(n_lines))
    
    ! --- Robust Data Ingestion Loop ---
    do i = 1, n_lines
        ! Attempt explicit formatted read for high-precision scientific notation
        read(unit, *, iostat=ios) lond, latd, velu, velv, bear, velo, dir, antenna_flag
        
        if (ios /= 0) then
            ! Fallback to list-directed (free-format) input if explicit format fails
            backspace(unit)
            read(unit, *, iostat=ios) lond, latd, velu, velv, bear, velo, dir, antenna_flag
            
            if (ios /= 0) then
                print *, 'Error: Critical failure at line ', i, ' iostat=', ios
                backspace(unit)
                read(unit, '(A)') line_buffer
                print *, 'Offending record: ', trim(line_buffer)
                close(unit)
                return
            end if
        end if
        
        ! Assign parsed values to the RadialData structure
        this%radials%lond(i) = lond
        this%radials%latd(i) = latd
        this%radials%velu(i) = velu
        this%radials%velv(i) = velv
        this%radials%bear(i) = bear
        this%radials%velo(i) = velo
        this%radials%dir(i) = dir
        this%radials%antenna_flag(i) = nint(antenna_flag) ! Map to nearest integer ID
        this%radials%bear_rad(i) = dir * PI / 180.0d0     ! Angular conversion to radians
        
        ! Debug: Output initial records to verify ingestion accuracy
        if (i <= 3) then
            print '(A,I0,A,8ES15.6)', 'Record ', i, ' parsed:', &
                  lond, latd, velu, velv, bear, velo, dir, real(antenna_flag)
        end if
    end do
    
    close(unit)
    
    ! --- Unique Sensor Identification ---
    allocate(temp_flags(n_lines))
    n_unique = 0
    
    do i = 1, n_lines
        flag_exists = .false.
        do j = 1, n_unique
            if (temp_flags(j) == this%radials%antenna_flag(i)) then
                flag_exists = .true.
                exit
            end if
        end do
        if (.not. flag_exists) then
            n_unique = n_unique + 1
            temp_flags(n_unique) = this%radials%antenna_flag(i)
        end if
    end do
    
    ! --- Site Metadata Construction ---
    this%n_sites = n_unique
    allocate(this%sites(n_unique))
    allocate(unique_flags(n_unique))
    unique_flags(1:n_unique) = temp_flags(1:n_unique)
    
    do i = 1, n_unique
        sum_lat = 0.0d0
        sum_lon = 0.0d0
        count_flag = 0
        
        ! Calculate mean geographic center for each unique antenna site
        do j = 1, n_lines
            if (this%radials%antenna_flag(j) == unique_flags(i)) then
                sum_lat = sum_lat + this%radials%latd(j)
                sum_lon = sum_lon + this%radials%lond(j)
                count_flag = count_flag + 1
            end if
        end do
        
        write(this%sites(i)%name, '(A,I0)') 'ANTENNA_', unique_flags(i)
        this%sites(i)%latitude = sum_lat / count_flag
        this%sites(i)%longitude = sum_lon / count_flag
        this%sites(i)%flag = unique_flags(i)
        
        print '(A,I0,A,F10.6,A,F10.6)', '  Antenna ', unique_flags(i), &
              ': Mean Lat=', this%sites(i)%latitude, ' Mean Lon=', this%sites(i)%longitude
    end do
    
    print '(A,I0)', '  Total sites detected: ', n_unique
    print '(A,I0)', '  Valid radial vectors: ', n_lines
    
    deallocate(temp_flags, unique_flags)
    success = .true.
    end subroutine parse_text_file

    !===========================================================================
    ! Subroutine: read_grid_from_netcdf
    ! Description: Ingests the spatial computational grid from a NetCDF file.
    !===========================================================================
    subroutine read_grid_from_netcdf(this, grid_file, LON, LAT, grid_lons, grid_lats, &
                                     mask_antena, n_lon, n_lat, n_ant, success)
        use netcdf
        class(HFRadarProcessor), intent(inout) :: this
        character(len=*), intent(in) :: grid_file
        real(8), allocatable, intent(out) :: LON(:), LAT(:)
        real(8), allocatable, intent(out) :: grid_lons(:,:), grid_lats(:,:)
        real(8), allocatable, intent(out) :: mask_antena(:,:,:)
        integer, intent(out) :: n_lon, n_lat, n_ant
        logical, intent(out) :: success
        
        integer :: ncid, varid, dimid
        integer :: i, j, status
        
        success = .false.
        print *, 'Loading computational grid from: ', trim(grid_file)
        
        ! Open NetCDF file in read-only mode
        status = nf90_open(trim(grid_file), NF90_NOWRITE, ncid)
        if (status /= NF90_NOERR) then
            print *, 'Error: Failed to open NetCDF grid file'
            return
        end if
        
        ! Retrieve dimension handles and lengths
        status = nf90_inq_dimid(ncid, 'lon', dimid)
        if (status /= NF90_NOERR) then
            print *, 'Error: Dimension "lon" not found'
            status = nf90_close(ncid)
            return
        end if
        status = nf90_inquire_dimension(ncid, dimid, len=n_lon)
        
        status = nf90_inq_dimid(ncid, 'lat', dimid)
        if (status /= NF90_NOERR) then
            print *, 'Error: Dimension "lat" not found'
            status = nf90_close(ncid)
            return
        end if
        status = nf90_inquire_dimension(ncid, dimid, len=n_lat)
        
        ! Antenna dimension (optional)
        status = nf90_inq_dimid(ncid, 'antenna', dimid)
        if (status == NF90_NOERR) then
            status = nf90_inquire_dimension(ncid, dimid, len=n_ant)
        else
            n_ant = 0
        end if
        
        ! Allocate grid arrays
        allocate(LON(n_lon))
        allocate(LAT(n_lat))
        allocate(grid_lons(n_lat, n_lon))
        allocate(grid_lats(n_lat, n_lon))
        
        ! Ingest coordinate variables
        status = nf90_inq_varid(ncid, 'lon', varid)
        status = nf90_get_var(ncid, varid, LON)
        
        status = nf90_inq_varid(ncid, 'lat', varid)
        status = nf90_get_var(ncid, varid, LAT)
        
        ! Ingest spatial mask if available
        if (n_ant > 0) then
            allocate(mask_antena(n_lat, n_lon, n_ant))
            status = nf90_inq_varid(ncid, 'mask', varid)
            if (status == NF90_NOERR) then
                status = nf90_get_var(ncid, varid, mask_antena)
            else
                mask_antena = 1.0d0  ! Default to no mask (unconstrained)
            end if
        else
            allocate(mask_antena(1, 1, 1))
            mask_antena = 1.0d0
        end if
        
        status = nf90_close(ncid)
        
        ! Construct 2D coordinate meshgrids
        do i = 1, n_lat
            do j = 1, n_lon
                grid_lons(i, j) = LON(j)
                grid_lats(i, j) = LAT(i)
            end do
        end do
        
        print '(A,F8.3,A,F8.3)', 'Latitude Extent:  ', minval(LAT), ' to ', maxval(LAT)
        print '(A,F8.3,A,F8.3)', 'Longitude Extent: ', minval(LON), ' to ', maxval(LON)
        
        success = .true.
    end subroutine read_grid_from_netcdf

    !===========================================================================
    ! Function: calculate_distance
    ! Description: Computes the great-circle distance between two geographic 
    !              points using the Haversine formula.
    !===========================================================================
    function calculate_distance(this, lat1, lon1, lat2, lon2) result(distance)
        class(HFRadarProcessor), intent(in) :: this
        real(8), intent(in) :: lat1, lon1, lat2, lon2
        real(8) :: distance
        
        real(8) :: lat1_rad, lat2_rad, dlat, dlon, a, c
        
        ! Convert coordinates to radians
        lat1_rad = lat1 * PI / 180.0d0
        lat2_rad = lat2 * PI / 180.0d0
        dlat = (lat2 - lat1) * PI / 180.0d0
        dlon = (lon2 - lon1) * PI / 180.0d0
        
        ! Haversine implementation
        a = sin(dlat/2.0d0)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2.0d0)**2
        c = 2.0d0 * asin(sqrt(a))
        
        distance = EARTH_RADIUS * c
    end function calculate_distance

    !===========================================================================
    ! Subroutine: least_squares_combination
    ! Description: Performs Unweighted Least Squares (ULS) to synthesize total 
    !              velocity vectors from nearby radial observations.
    !===========================================================================
    subroutine least_squares_combination(this, grid_lons, grid_lats, n_lat, n_lon, &
                                        max_distance, min_observations, results, n_results)
        class(HFRadarProcessor), intent(in) :: this
        integer, intent(in) :: n_lat, n_lon
        real(8), intent(in) :: grid_lons(n_lat, n_lon)
        real(8), intent(in) :: grid_lats(n_lat, n_lon)
        real(8), intent(in) :: max_distance
        integer, intent(in) :: min_observations
        type(TotalVelocity), allocatable, intent(out) :: results(:)
        integer, intent(out) :: n_results
        
        integer :: i, j, k, n_obs
        real(8) :: grid_lat, grid_lon, distance
        real(8), allocatable :: obs_matrix(:,:)
        integer, allocatable :: antennas_rep(:)
        integer :: n_antennas
        real(8), allocatable :: A(:,:), ATA(:,:), C(:,:), ATb(:)
        real(8), allocatable :: radial_vels(:), bearings(:), alpha(:)
        real(8) :: velocity(2), u_vel, v_vel, vel_mag, vel_dir, gdop
        integer :: total_points, processed
        real(8) :: det
        type(TotalVelocity), allocatable :: temp_results(:)
        integer :: max_results
        
        call system_clock(t_start, count_rate)
        max_results = n_lat * n_lon
        allocate(temp_results(max_results))
        n_results = 0
        total_points = n_lat * n_lon
        processed = 0
        
        print '(A,I0,A)', 'Synthesizing vectors for ', total_points, ' grid points...'
        
        do i = 1, n_lat
            do j = 1, n_lon
                grid_lat = grid_lats(i, j)
                grid_lon = grid_lons(i, j)
                
                processed = processed + 1
                if (mod(processed, 1000) == 0) then
                    print '(A,I0,A,I0,A)', '  Progress: ', processed, '/', total_points, ' nodes...'
                    call system_clock(t_end)
                    elapsed = real(t_end - t_start, 8) / real(count_rate, 8)
                    write(*,'(A, F12.6, A)') '  CPU Time: ', elapsed, ' seconds'
                end if
                
                ! --- Neighbor Search: Locate radials within max_distance radius ---
                n_obs = 0
                allocate(obs_matrix(this%radials%n_points, 3))
                allocate(antennas_rep(this%radials%n_points))
                n_antennas = 0
                
                do k = 1, this%radials%n_points
                    distance = this%calculate_distance(grid_lat, grid_lon, &
                                                       this%radials%latd(k), &
                                                       this%radials%lond(k))
                    
                    if (distance <= max_distance) then
                        n_obs = n_obs + 1
                        obs_matrix(n_obs, 1) = this%radials%velo(k)
                        obs_matrix(n_obs, 2) = this%radials%bear_rad(k)
                        obs_matrix(n_obs, 3) = distance
                        
                        ! Identify unique site contributions within the radius
                        if (.not. any(antennas_rep(1:n_antennas) == this%radials%antenna_flag(k))) then
                            n_antennas = n_antennas + 1
                            antennas_rep(n_antennas) = this%radials%antenna_flag(k)
                        end if
                    end if
                end do
                
                ! --- Numerical Synthesis Requirements ---
                ! Minimum 2 unique sites and specified observation count required
                if (n_antennas >= 2 .and. n_obs >= min_observations) then
                    
                    allocate(radial_vels(n_obs), bearings(n_obs), alpha(n_obs))
                    allocate(A(n_obs, 2), ATA(2, 2), C(2, 2), ATb(2))
                    
                    radial_vels = obs_matrix(1:n_obs, 1)
                    bearings = obs_matrix(1:n_obs, 2) * 180.0d0 / PI
                    
                    ! Convert geographic bearings to mathematical trigonometric angles
                    alpha = mod(450.0d0 - bearings, 360.0d0) * PI / 180.0d0
                    
                    ! Construct Design Matrix A
                    A(:, 1) = cos(alpha)
                    A(:, 2) = sin(alpha)
                    
                    ! Form Normal Equations: (A^T * A) * x = A^T * b
                    ATA = matmul(transpose(A), A)
                    ATb = matmul(transpose(A), radial_vels)
                    
                    ! Solve via matrix inversion for 2x2 system
                    det = ATA(1,1) * ATA(2,2) - ATA(1,2) * ATA(2,1)
                    
                    if (abs(det) > 1.0d-10) then
                        ! Inverse of 2x2 matrix
                        C(1,1) = ATA(2,2) / det
                        C(1,2) = -ATA(1,2) / det
                        C(2,1) = -ATA(2,1) / det
                        C(2,2) = ATA(1,1) / det
                        
                        ! Resultant Total Velocity Vector
                        velocity = matmul(C, ATb)
                        u_vel = velocity(1)
                        v_vel = velocity(2)
                    
                        vel_mag = sqrt(u_vel**2 + v_vel**2)
                        vel_dir = atan2(u_vel, v_vel) * 180.0d0 / PI
                        if (vel_dir < 0.0d0) vel_dir = vel_dir + 360.0d0
                        
                        ! Calculate GDOP (Geometrical Dilution of Precision)
                        gdop = sqrt(abs(C(1,1) + C(2,2)))
                        
                        ! Log valid results to temporary buffer
                        n_results = n_results + 1
                        temp_results(n_results)%longitude = grid_lon
                        temp_results(n_results)%latitude = grid_lat
                        temp_results(n_results)%u_total = u_vel
                        temp_results(n_results)%v_total = v_vel
                        temp_results(n_results)%speed = vel_mag
                        temp_results(n_results)%direction = vel_dir
                        temp_results(n_results)%gdop = gdop
                        temp_results(n_results)%n_obs = n_obs
                        temp_results(n_results)%n_sites = n_antennas
                    end if
                
                    deallocate(radial_vels, bearings, alpha, A, ATA, C, ATb)
                end if
                deallocate(obs_matrix, antennas_rep)
            end do
        end do
        
        print '(A,I0,A)', 'Task complete. ', n_results, ' valid total vectors synthesized.'
        
        ! Transfer to permanent results array
        allocate(results(n_results))
        results(1:n_results) = temp_results(1:n_results)
        deallocate(temp_results)
        
    end subroutine least_squares_combination

    !===========================================================================
    ! Subroutine: write_results_txt
    ! Description: Serializes processing results to a structured ASCII file.
    !===========================================================================
    subroutine write_results_txt(this, output_path, results, n_results)
        class(HFRadarProcessor), intent(in) :: this
        character(len=*), intent(in) :: output_path
        type(TotalVelocity), intent(in) :: results(:)
        integer, intent(in) :: n_results
        
        integer :: unit, i
        character(len=19) :: datetime_str
        
        open(newunit=unit, file=trim(output_path), status='replace', action='write')
        
        ! Write metadata header
        call get_datetime(datetime_str)
        write(unit, '(A)') '# HF Radar Total Velocities - Calculated via Least Squares'
        write(unit, '(A,A)') '# Generation Timestamp: ', datetime_str
        write(unit, '(A,I0)') '# Total Vector Count:   ', n_results
        write(unit, '(A)') '#'
        write(unit, '(A)') '# Column Definitions:'
        write(unit, '(A)') '# longitude (deg) - Grid point longitude'
        write(unit, '(A)') '# latitude (deg)  - Grid point latitude'
        write(unit, '(A)') '# u_total (cm/s)  - Eastward velocity component'
        write(unit, '(A)') '# v_total (cm/s)  - Northward velocity component'
        write(unit, '(A)') '# speed (cm/s)    - Current magnitude'
        write(unit, '(A)') '# direction (deg) - Heading direction (0=N, 90=E)'
        write(unit, '(A)') '# gdop            - Geometrical Dilution of Precision'
        write(unit, '(A)') '#'
        write(unit, '(A)') 'longitude	latitude	u_total	v_total	speed	direction	gdop'
        
        ! Write data records
        do i = 1, n_results
            write(unit, '(F12.6,A,F12.6,A,F10.3,A,F10.3,A,F10.3,A,F10.1,A,F10.1)') &
                results(i)%longitude, '	', &
                results(i)%latitude, '	', &
                results(i)%u_total, '	', &
                results(i)%v_total, '	', &
                results(i)%speed, '	', &
                results(i)%direction, '	', &
                results(i)%gdop
        end do
        
        close(unit)
        print *, 'Output serialized to: ', trim(output_path)
    end subroutine write_results_txt

    !===========================================================================
    ! Subroutine: get_datetime
    ! Description: Formats current system date and time for file metadata.
    !===========================================================================
    subroutine get_datetime(datetime_str)
        character(len=19), intent(out) :: datetime_str
        integer :: values(8)
        
        call date_and_time(values=values)
        write(datetime_str, '(I4,A,I2.2,A,I2.2,A,I2.2,A,I2.2,A,I2.2)') &
            values(1), '-', values(2), '-', values(3), ' ', &
            values(5), ':', values(6), ':', values(7)
    end subroutine get_datetime

    !===========================================================================
    ! Subroutine: process_text_files
    ! Description: High-level workflow orchestrator. Manages ingestion, 
    !              computation, statistical analysis, and output generation.
    !===========================================================================
    subroutine process_text_files(this, input_file, output_file, grid_file, max_distance)
        class(HFRadarProcessor), intent(inout) :: this
        character(len=*), intent(in) :: input_file
        character(len=*), intent(in) :: output_file
        character(len=*), intent(in) :: grid_file
        real(8), intent(in) :: max_distance
        
        logical :: success
        real(8), allocatable :: LON(:), LAT(:), grid_lons(:,:), grid_lats(:,:)
        real(8), allocatable :: mask_antena(:,:,:)
        integer :: n_lon, n_lat, n_ant
        type(TotalVelocity), allocatable :: results(:)
        integer :: n_results
        real(8) :: mean_vel, max_vel, min_vel, std_vel, mean_obs, mean_sites
        real(8) :: sum_vel, sum_sq, variance
        integer :: i
        
        print *, '=== HF Radar Total Velocity Processor ==='
        print *, 'Input Path:  ', trim(input_file)
        print *, 'Output Path: ', trim(output_file)
        print '(A,F6.2,A)', 'Interpolation Radius: ', max_distance, ' km'
        print *
        
        ! Step 1: Ingest Radial Data
        print *, 'Parsing input radial file...'
        call this%parse_text_file(input_file, success)
        
        if (.not. success .or. this%radials%n_points == 0) then
            print *, 'Error: Ingestion failed. No valid radial data extracted.'
            return
        end if
        
        print *, 'Valid dataset identified.'
        print '(A,I0)', 'Radial vector count: ', this%radials%n_points
        
        ! Step 2: Load Computational Grid
        print *
        call this%read_grid_from_netcdf(grid_file, LON, LAT, grid_lons, grid_lats, &
                                       mask_antena, n_lon, n_lat, n_ant, success)
        
        if (.not. success) then
            print *, 'Error: Computational grid ingestion failed.'
            return
        end if
        
        ! Step 3: Perform Vector Synthesis
        print *
        print *, 'Initializing Least Squares Synthesis...'
        call this%least_squares_combination(grid_lons, grid_lats, n_lat, n_lon, &
                                           max_distance, 2, results, n_results)
        
        if (n_results == 0) then
            print *, 'Error: Zero total vectors generated. Verify search radius.'
            return
        end if
        
        ! Step 4: Statistical Evaluation
        sum_vel = 0.0d0
        sum_sq = 0.0d0
        max_vel = results(1)%speed
        min_vel = results(1)%speed
        mean_obs = 0.0d0
        mean_sites = 0.0d0
        
        do i = 1, n_results
            sum_vel = sum_vel + results(i)%speed
            sum_sq = sum_sq + results(i)%speed**2
            if (results(i)%speed > max_vel) max_vel = results(i)%speed
            if (results(i)%speed < min_vel) min_vel = results(i)%speed
            mean_obs = mean_obs + real(results(i)%n_obs)
            mean_sites = mean_sites + real(results(i)%n_sites)
        end do
        
        mean_vel = sum_vel / n_results
        variance = (sum_sq / n_results) - mean_vel**2
        std_vel = sqrt(variance)
        mean_obs = mean_obs / n_results
        mean_sites = mean_sites / n_results
        
        print *
        print *, '=== Execution Statistics ==='
        print '(A,I0)',   'Valid Total Vectors:          ', n_results
        print '(A,F8.2,A)', 'Mean Velocity Magnitude:      ', mean_vel, ' units'
        print '(A,F8.2,A)', 'Maximum Velocity Magnitude:   ', max_vel, ' units'
        print '(A,F8.2,A)', 'Minimum Velocity Magnitude:   ', min_vel, ' units'
        print '(A,F8.2,A)', 'Velocity Standard Deviation:  ', std_vel, ' units'
        print '(A,F6.1)',   'Mean Observations per Point:  ', mean_obs
        print '(A,F6.1)',   'Mean Sensors per Point:       ', mean_sites
        
        ! Step 5: Persistence
        print *
        print *, 'Exporting results...'
        call this%write_results_txt(output_file, results, n_results)
        
        print *
        print *, 'Processing lifecycle completed successfully.'
        
        ! Cleanup local scope allocations
        deallocate(LON, LAT, grid_lons, grid_lats, mask_antena, results)
    end subroutine process_text_files

    !===========================================================================
    ! Subroutine: cleanup
    ! Description: Explicitly releases all dynamically allocated memory within 
    !              the class instance.
    !===========================================================================
    subroutine cleanup(this)
        class(HFRadarProcessor), intent(inout) :: this
        
        if (allocated(this%radials%lond)) deallocate(this%radials%lond)
        if (allocated(this%radials%latd)) deallocate(this%radials%latd)
        if (allocated(this%radials%velu)) deallocate(this%radials%velu)
        if (allocated(this%radials%velv)) deallocate(this%radials%velv)
        if (allocated(this%radials%bear)) deallocate(this%radials%bear)
        if (allocated(this%radials%velo)) deallocate(this%radials%velo)
        if (allocated(this%radials%dir)) deallocate(this%radials%dir)
        if (allocated(this%radials%antenna_flag)) deallocate(this%radials%antenna_flag)
        if (allocated(this%radials%bear_rad)) deallocate(this%radials%bear_rad)
        if (allocated(this%sites)) deallocate(this%sites)
    end subroutine cleanup
    
end module HFradar_data
