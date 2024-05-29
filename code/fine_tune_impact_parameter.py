"""
This module contains functions that are used to fine-tune the impact parameter of the stream. 
The first estiamte was used by tracking the magnitude of the gravitational force of the globular clusters on the orbit.
This gave us a good estimate of when the impact happened, but not the exact position of the impact along the stream, not the exact moment in time. 
This is because (1) the stream is offset from the orbit and (2) the stream was not saved at high temporal resolution. 
Therefore, this module is used to perform a parametric fit of the stream and perturber trajectory to find the exact impact position and time analytically. 


Example:
>>> mcarlo          = "monte-carlo-009"
>>> perturberName   = "NGC104"
>>> pathStreamOrbit = ""
>>> pathPerturOrbit = ""
>>> pathPal5Orbit   = ""
>>> pathForceFile   = ""
>>> streamOrbit     =   h5py.File(pathStreamOrbit, "r")
>>> perturberOrbit  =   h5py.File(pathPerturOrbit, "r")
>>> pal5Orbit       =   h5py.File(pathPal5Orbit, "r")
>>> forceFile       =   h5py.File(pathForceFile, "r")
>>> temporal_coefficients_array,stream_coordinate_range,simulation_sampling_time_stamps=\
>>>         get_temporal_coefficients_array_of_parametric_3D_stream(\
>>>         mcarlo,perturberName,streamOrbit,perturberOrbit,pal5Orbit,forceFile,)
>>> endtime=datetime.datetime.now()
>>> print("Time to perform stream fit",endtime-starttime)
>>> # get the linearized coefficients, how the coefficients change with time
>>> coefficient_time_fit_params = linearize_temporal_stream_coefficients_array(temporal_coefficients_array,simulation_sampling_time_stamps)
>>> # get the coefficients for a polynomial describign the's perturber orbit
>>> time_range = np.array([simulation_sampling_time_stamps[0].value,simulation_sampling_time_stamps[-1].value])
>>> trajectory_coeffs=parameterize_oribtal_trajectory(perturberOrbit,mcarlo,time_range)
>>> # make the initial guess for the impact position and time
>>> t0=np.mean(simulation_sampling_time_stamps.value)
>>> s0=np.mean(stream_coordinate_range)
>>> p0=(s0,t0)
>>> # find the impact position and time
>>> minimization_method='Nelder-Mead'
>>> results=optimize.minimize(
>>>     PSF.objective_distance_perturber_and_moving_stream,
>>>     p0,
>>>     args=(coefficient_time_fit_params,trajectory_coeffs),
>>>     method=minimization_method,)
>>> s,t=results.x
>>> # Get the impact geometry for erkal 2015
>>> rs,vs,rp,b_vec_galactic,b_vec_tail_basis,impact_parameter,w_par,w_per,alpha=\
>>>     get_full_impact_geometry_from_parametrization(s,t,coefficient_time_fit_params,trajectory_coeffs)

...
Author: Salvatore Ferrone
Date: April 2024 (first functioning full version)
"""
import numpy as np
import astropy.units as u
import h5py
from scipy import optimize
## My modules
import parametric_stream_fitting as PSF
import data_extractors as DE 
# one day I need to make this into a good module that can be imported for the whole environment
import sys
codepath="/obs/sferrone/gc-tidal-loss/code/"
sys.path.append(codepath)
import StreamOrbitCoords as SOC

#######################################################
######## High level functions with data inputs ########
#######################################################

def get_temporal_coefficients_array_of_parametric_3D_stream(\
    mcarlo,
    perturberName,
    streamOrbit,
    perturberOrbit,
    pal5Orbit,
    forceFile,
):
    """
    VERY HIGH LEVEL
    
    a very high level code intended to fit the stream to a 3D parabola at some instances before and after the impact.
    """ 
    
    # parameters that will become function arguments with default values
    xmin,xlim,ylim,zlim=0.5,12,0.5,0.5
    # the number of time stamps adjacent to the impact time
    n_adjacent_points=2
    
    ### non changing parameters
    unitT=u.kpc/(u.km/u.s)
    minimization_method='Nelder-Mead'
    

    
    #### Obtain the data at the time of impact
    filter, \
    stream_time_coordinate, stream_tail_coordinates,stream_galactic_coordinates, \
    perturber_time_coordinate, perturber_tail_coordinates,perturber_galactic_coordinates,\
    host_orbit_galactic_coordinates,\
    index_simulation_time_of_impact, index_stream_orbit_time_of_impact,\
    impact_time=\
    obtain_star_particle_filter_from_data_files(\
        forceFile,streamOrbit,perturberOrbit,pal5Orbit,\
        mcarlo,perturberName,\
        xmin,xlim,ylim,zlim)
    
    
    # the time range of the stream coordinate    
    tmin=stream_time_coordinate[filter].min() - (1/10)*np.abs(stream_time_coordinate[filter].min())
    tmax=stream_time_coordinate[filter].max() + (1/10)*np.abs(stream_time_coordinate[filter].max())
    stream_coordinate_range=[tmin,tmax]
    ####### BEGIN GETTING THE FIT PARAMETERS FOR THE CURVES ########
    # get the initial guess
    initial_guess=PSF.constrain_3D_parabola_initial_guess(
        stream_time_coordinate[filter],
        stream_galactic_coordinates[0:3,
        filter])
    # DO THE FITTING
    results=optimize.minimize(
        PSF.objective_parametric_3D_parabola,
        initial_guess,
        args=(stream_time_coordinate[filter],stream_galactic_coordinates[0:3,filter]),
        method=minimization_method,)
    
    ##### Start storing the output results
    n_stream_samplings = 1 + 2*n_adjacent_points
    center_index = n_stream_samplings//2 
    temporal_coefficients_array=np.zeros((n_stream_samplings,len(results.x)))
    temporal_coefficients_array[center_index,:]=results.x
    
    # get the stream orbit time stamps, will need these again
    stream_orbit_time_stamps=DE.extract_time_steps_from_stream_orbit(streamOrbit)
    
    # Do analysis for the adjacent time stamps. Delete the center index to not repeate
    stream_indexes_of_interest = np.arange(
        index_stream_orbit_time_of_impact-n_adjacent_points,
        index_stream_orbit_time_of_impact+n_adjacent_points+1)
    # grab the simulation time stamps
    simulation_sampling_time_stamps=stream_orbit_time_stamps[stream_indexes_of_interest]
    
    ii=0
    for new_index_of_interest_stream_orbit in stream_indexes_of_interest:
        if new_index_of_interest_stream_orbit==index_stream_orbit_time_of_impact:
            # don't repeat the center index
            ii+=1
            continue
        else:
            new_index_of_interest_simulation=np.argmin(np.abs(pal5Orbit[mcarlo]['t'][:]-stream_orbit_time_stamps[new_index_of_interest_stream_orbit].value))
            # extract the stream
            stream_galactic_coordinates = DE.get_galactic_coordinates_of_stream(new_index_of_interest_stream_orbit,streamOrbit)
            # apply the filter to the stream to only obtain particles of interest
            stream_subset_galactic_coordinates=stream_galactic_coordinates[:,filter]
            # extract host orbit and perturber position of this time
            host_orbit_galactic_coordinates, perturber_galactic_coordinates = \
                DE.get_galactic_coordinates_of_host_orbit_and_perturber(
                    new_index_of_interest_simulation,
                    pal5Orbit                           =       pal5Orbit,
                    perturberOrbit                      =       perturberOrbit,
                    monte_carlo_key                     =       mcarlo)
            # grab new time of interest
            time_of_interset=pal5Orbit[mcarlo]['t'][index_simulation_time_of_impact]
            # get the tail coordinates. Notice that the time will already be filtered
            stream_subset_time_coordinate,_,_,_=\
                convert_instant_to_tail_coordinates(\
                    stream_subset_galactic_coordinates,
                    host_orbit_galactic_coordinates,
                    perturber_galactic_coordinates,
                    time_of_interset)
            # get the initial guess
            p0=PSF.constrain_3D_parabola_initial_guess(stream_subset_time_coordinate,stream_subset_galactic_coordinates[0:3,:])
            # perform the fit
            results=optimize.minimize(PSF.objective_parametric_3D_parabola,p0,args=(stream_subset_time_coordinate,stream_subset_galactic_coordinates[0:3,:]),method=minimization_method,)
            temporal_coefficients_array[ii,:]=results.x
            ii+=1
    
    
    return temporal_coefficients_array,stream_coordinate_range,simulation_sampling_time_stamps
    
def parameterize_oribtal_trajectory(\
    orbit_file: h5py.File, \
    mcarlo: str, \
    simulation_time_range:np.ndarray) -> np.ndarray:
    """
    Parameterizes the orbital trajectory using polynomial fitting.

    Parameters:
    orbit_file (h5py.File): The HDF5 file containing the orbital data.
    mcarlo (str): The name of the Monte Carlo simulation.
    simulation_time_range (np.ndarray): The time range for the simulation.

    Returns:
    trajectory_coeffs (np.ndarray): The coefficients of the polynomial fit for each dimension of the trajectory.
    
    Example: 
    >>> trajectory_coeffs=parameterize_oribtal_trajectory(orbit_file, mcarlo="monte-carlo-009", simulation_time_range=np.array([-0.14, -0.10])
    >>> # extract the coefficients
    >>> alpha0, alpha1, alpha2 = trajectory_coeffs[0,:]
    >>> beta0, beta1, beta2 = trajectory_coeffs[1,:]
    >>> gamma0, gamma1, gamma2 = trajectory_coeffs[2,:]
    >>> # get the x,y,z coordinates at time t
    >>> x_t = alpha0 + alpha1*t + alpha2*t^2
    >>> y_t = beta0 + beta1*t + beta2*t^2
    >>> z_t = gamma0 + gamma1*t + gamma2*t^2
    """

    polynomial_degree = 2
    
    simulation_initial_time_step=np.argmin(np.abs(orbit_file[mcarlo]['t'][:] - simulation_time_range[0]))
    simulation_end_time_step=np.argmin(np.abs(orbit_file[mcarlo]['t'][:] - simulation_time_range[-1]))


    t_simulation = orbit_file[mcarlo]['t'][simulation_initial_time_step:simulation_end_time_step]
    xt_pert = orbit_file[mcarlo]['xt'][simulation_initial_time_step:simulation_end_time_step]
    yt_pert = orbit_file[mcarlo]['yt'][simulation_initial_time_step:simulation_end_time_step]
    zt_pert = orbit_file[mcarlo]['zt'][simulation_initial_time_step:simulation_end_time_step]
    
    trajectory_coeffs = np.zeros((3,polynomial_degree+1))

    trajectory_coeffs[0,:]=np.polyfit(t_simulation,xt_pert,polynomial_degree)
    trajectory_coeffs[1,:]=np.polyfit(t_simulation,yt_pert,polynomial_degree)
    trajectory_coeffs[2,:]=np.polyfit(t_simulation,zt_pert,polynomial_degree)
    
    return trajectory_coeffs
 
def obtain_star_particle_filter_from_data_files(
    forceFile: h5py.File,
    streamOrbit: h5py.File,
    perturberOrbit: h5py.File,
    pal5Orbit: h5py.File,
    mcarlo: str,
    perturberName: str,
    xmin:float,
    xlim:float,
    ylim:float,
    zlim:float
):
    """
    This is a high level function. Extracts the filter from the data files based on the given parameters.

    Parameters:
    - forceFile (h5py.File): The force file containing information about the impact.
    - streamOrbit (h5py.File): The file containing the orbit of the stream.
    - perturberOrbit (h5py.File): The file containing the orbit of the perturber.
    - pal5Orbit (h5py.File): The file containing the orbit of the host (Pal 5).
    - mcarlo (str): The Monte Carlo key.
    - perturberName (str): The name of the perturber.
    - xmin (float): The minimum value for the x-coordinate.
    - xlim (float): The maximum value for the x-coordinate.
    - ylim (float): The maximum value for the y-coordinate.
    - zlim (float): The maximum value for the z-coordinate.

    Returns:
    - filter (numpy.ndarray): The filter representing the particles on the same side as the Galactic Center.
    - stream_time_coordinate (numpy.ndarray): The time coordinate of the stream. coordinate along orbit of particle ahead of behind the cluster
    - stream_tail_coordinates (numpy.ndarray): The tail coordinates of the stream.
    - perturber_time_coordinate (numpy.ndarray): The time coordinate of the perturber.
    - perturber_tail_coordinates (numpy.ndarray): The tail coordinates of the perturber.
    """

    # get indicies of when impact happened
    index_simulation_time_of_impact, index_stream_orbit_time_of_impact = DE.coordinate_impact_indicies_from_force_file(
        forceFile, streamOrbit, perturberOrbit,monte_carlo_key=mcarlo,perturberName=perturberName)
    # extract the stream at this time
    stream_galactic_coordinates = DE.get_galactic_coordinates_of_stream(index_stream_orbit_time_of_impact,streamOrbit)
    # extract host orbit and perturber position of this time
    host_orbit_galactic_coordinates, perturber_galactic_coordinates = DE.get_galactic_coordinates_of_host_orbit_and_perturber(
        index_simulation_time_of_impact,pal5Orbit=pal5Orbit,perturberOrbit=perturberOrbit,monte_carlo_key=mcarlo)
    # get impact time, in units of time of the simulation
    impact_time=pal5Orbit[mcarlo]['t'][index_simulation_time_of_impact]
    # convert the stream and perturber to tail coordinates
    stream_time_coordinate,stream_tail_coordinates,perturber_time_coordinate,perturber_tail_coordinates=\
        convert_instant_to_tail_coordinates(stream_galactic_coordinates,host_orbit_galactic_coordinates,perturber_galactic_coordinates,impact_time)
    # filter to only take the particles on the same side as the GC
    filter = filter_stream_about_suspected_impact_time(stream_tail_coordinates,perturber_tail_coordinates,xmin,xlim,ylim,zlim)
    
    return filter, stream_time_coordinate, stream_tail_coordinates,stream_galactic_coordinates, perturber_time_coordinate, perturber_tail_coordinates,perturber_galactic_coordinates,host_orbit_galactic_coordinates,index_simulation_time_of_impact, index_stream_orbit_time_of_impact,impact_time

##############################################################
########### Functions that perform computations only. ########
##############################################################

def get_full_impact_geometry_from_parametrization(
    s:float,
    t:float,
    coefficient_time_fit_params: np.ndarray,
    trajectory_coeffs: np.ndarray,):
    
    # get the stream position and velocity
    rs = PSF.moving_stream_parametric_3D_parabola(s,t,coefficient_time_fit_params)
    vs = PSF.velocity_moving_stream_parametric_3D_parabola(s,t,coefficient_time_fit_params)
    # get the tail coordinate basis vectors, a quasi-orthonormal basis
    r_unit, v_unit, Lz_unit=get_tail_basis_vectors(rs,vs)
    # get the position of the perturber
    rp = get_perturber_position(t,trajectory_coeffs)
    # get the impact parameter as a vector
    b_vec_galactic=rp-rs
    b_vec_tail_basis=get_impact_vector_in_tail_basis(b_vec_galactic,r_unit,v_unit,Lz_unit)
    # the magnitude of b_vec should be the impact impact_parameter
    impact_parameter=PSF.objective_distance_perturber_and_moving_stream((s,t),coefficient_time_fit_params,trajectory_coeffs)
    w_par,w_per=get_parallel_and_perpendicular_velocity_components(s,t,trajectory_coeffs,coefficient_time_fit_params)
    alpha=get_imact_alpha_from_b_vector(b_vec_tail_basis)
    return impact_parameter,w_par,w_per,alpha,rs,vs,rp,b_vec_galactic,b_vec_tail_basis
    
def get_imact_alpha_from_b_vector(b_vec_tail_basis:np.ndarray):
    """
    Returns the impact parameter in the direction of the position vector.
    """
    tolerence=2e-2
    alpha1=np.arccos(b_vec_tail_basis[0]/np.linalg.norm(b_vec_tail_basis))
    alpha2=np.arcsin(b_vec_tail_basis[2]/np.linalg.norm(b_vec_tail_basis))
    alpha3=np.arctan2(b_vec_tail_basis[2],b_vec_tail_basis[0])
    to_degrees=180/np.pi
    mean_alpha=np.mean([alpha1,alpha2])
    discrepancy=np.abs(alpha1-alpha2) 
    if (discrepancy/mean_alpha)>tolerence:
        war_string = "Warning, the impact angles are highly inconsistent. Investigate. Alpha to x-axis: {:.3e}\n Complement to z-axis: {:.3e}\sDiscrepancy: {:.3e}".format(to_degrees*alpha1,to_degrees*alpha2,to_degrees*discrepancy)
        Warning(war_string)
    return alpha3

def get_impact_vector_in_tail_basis(b_vec_galactic:np.ndarray,
                     r_unit:np.ndarray,
                     v_unit:np.ndarray,
                     Lz_unit:np.ndarray) -> float:
    
    b_vec_tail_basis = np.array([np.dot(b_vec_galactic,r_unit),
                      np.dot(b_vec_galactic,v_unit),
                      np.dot(b_vec_galactic,Lz_unit)])
    return b_vec_tail_basis

def get_tail_basis_vectors(rs:np.ndarray,vs:np.ndarray):
    """
    Calculates the basis vectors for impact parameters based on the given parameters.

    Parameters:
    rs (np.ndarray): The point of impact along the stream in the galactic frame
    vs (np.ndarray): The the velocity of the stream at the point of impact

    Returns:
    tuple: A tuple containing the basis vectors (r_unit, v_unit, Lz_unit).
        r_unit (numpy.ndarray): The unit vector in the direction of the position vector.
        v_unit (numpy.ndarray): The unit vector in the direction of the velocity vector.
        Lz_unit (numpy.ndarray): The unit vector in the direction of the angular momentum vector.
        
    Example:
    >>> # Given s,t,coefficient_time_fit_params from the previous functions
    >>> rs = PSF.moving_stream_parametric_3D_parabola(s,t,coefficient_time_fit_params)
    >>> vs = PSF.velocity_moving_stream_parametric_3D_parabola(s,t,coefficient_time_fit_params)
    >>>  r_unit, v_unit, Lz_unit = get_impact_basis_vectors(rs, vs)
    """

    r_unit = rs / np.linalg.norm(rs)
    v_unit = vs / np.linalg.norm(vs)
    Lz_unit = np.cross(r_unit, v_unit) / np.linalg.norm(np.cross(r_unit, v_unit))

    return r_unit, v_unit, Lz_unit

def get_parallel_and_perpendicular_velocity_components(
    s:float,
    t:float,
    trajectory_coeffs: np.ndarray,
    stream_temporal_coefficients_array: np.ndarray,):
    """
    
    """
    
    vs =  PSF.velocity_moving_stream_parametric_3D_parabola(s,t,stream_temporal_coefficients_array)
    vp = get_perturber_velocity(t,trajectory_coeffs)
    
    unit_vector = vs/np.linalg.norm(vs)
    
    dv = vp-vs
    w_parallel = np.dot(dv,unit_vector)*unit_vector
    w_perpendicular = dv - w_parallel

    
    return w_parallel, w_perpendicular

def get_perturber_position(t:float,trajectory_coeffs: np.ndarray):
    """
    Assumes that we have a second degree polynomial in time for the trajectory.
    Also assumes that coefficients are in descending order of time, as in np.polyfit
    """
    assert trajectory_coeffs.shape[0]==3, "The trajectory coefficients must have shape (3, polynomial_degree+1)"
    assert trajectory_coeffs.shape[1]==3, "The trajectory must be a 2nd degree polynomial in time"
    xp,yp,zp = np.polyval(trajectory_coeffs[0],t),np.polyval(trajectory_coeffs[1],t),np.polyval(trajectory_coeffs[2],t)
    rp=np.array([xp,yp,zp])
    return rp

def get_perturber_velocity(t:float,trajectory_coeffs: np.ndarray):
    """
    Assumes that we have a second degree polynomial in time for the trajectory.
    Also assumes that coefficients are in descending order of time, as in np.polyfit
    
    """
    assert trajectory_coeffs.shape[0]==3, "The trajectory coefficients must have shape (3, 2+1), 2 for 2nd degree"
    assert trajectory_coeffs.shape[1]==3, "The trajectory must be a 2nd degree polynomial in time"
    
    vxp=2*trajectory_coeffs[0,0]*t + trajectory_coeffs[0,1]
    vyp=2*trajectory_coeffs[1,0]*t + trajectory_coeffs[1,1]
    vzp=2*trajectory_coeffs[2,0]*t + trajectory_coeffs[2,1]
    vp=np.array([vxp,vyp,vzp])
    return vp

def linearize_temporal_stream_coefficients_array(\
    temporal_coefficients_array: np.ndarray, \
    simulation_sampling_time_stamps: np.ndarray) -> np.ndarray:
    """
    Linearizes the temporal coefficients array.

    This function takes a temporal coefficients array and returns its linearized version.
    
    Parameters:
    - temporal_coefficients_array (np.ndarray): The input array containing temporal coefficients.
    - simulation_sampling_time_stamps (np.ndarray): The array of time stamps corresponding to the simulation sampling.

    Returns:
    - np.ndarray: The linearized version of the temporal coefficients array.

    Notes:
    - The input array should have shape (n, m), where n is the number of samples and m is the number of coefficients.
    - The output array will have shape (m, polynomial_degree+1), where polynomial_degree is the degree of the polynomial fit.

    Coefficient Time Fit Params:
    - coefficient_time_fit_params[0,:] = a0(t) = (m,b)
    - coefficient_time_fit_params[1,:] = a1(t) = (m,b)
    - coefficient_time_fit_params[2,:] = a2(t) = (m,b)
    - coefficient_time_fit_params[3,:] = b0(t) = (m,b)
    - coefficient_time_fit_params[4,:] = b1(t) = (m,b)
    - coefficient_time_fit_params[5,:] = b2(t) = (m,b)
    - coefficient_time_fit_params[6,:] = c0(t) = (m,b)
    - coefficient_time_fit_params[7,:] = c1(t) = (m,b)
    - coefficient_time_fit_params[8,:] = c2(t) = (m,b)
    """
    polynomial_degree = 1
    coefficient_time_fit_params = np.zeros((temporal_coefficients_array.shape[1], polynomial_degree + 1))
    for i in range(temporal_coefficients_array.shape[1]):
        coefficient_time_fit_params[i] = np.polyfit(simulation_sampling_time_stamps, temporal_coefficients_array[:, i], polynomial_degree)
    return coefficient_time_fit_params

def filter_stream_about_suspected_impact_time(stream_tail_coordinates:np.ndarray,
                                perturber_tail_coordinates:np.ndarray,
                                xmin:float=0.5,
                                xlim:float=15,
                                ylim:float=0.5,
                                zlim:float=0.5) -> np.ndarray:
    """
    Filters the stream to only include the one of the two tails. Pick the side the the GC is on. 
        Also, filter to only include a given range of the stream based on the xlim,ylim,zlim
        additionally, xmin above zero clips the stars that are still bound to the globular cluster

    Parameters:
    - stream_tail_coordinates (np.ndarray): The coordinates of the stream tail.
    - perturber_tail_coordinates (np.ndarray): The coordinates of the perturber tail.
    - xmin (float, optional): The minimum x-coordinate value. Defaults to 0.5. Intended to remove the globular cluster, only taking the stream.
    - xlim (float, optional): The maximum x-coordinate value. Defaults to 15.
    - ylim (float, optional): The maximum y-coordinate value. Defaults to 0.5.
    - zlim (float, optional): The maximum z-coordinate value. Defaults to 0.5.

    Returns:
    - filter (np.ndarray): A boolean array indicating which elements of the stream should be included.

    """
    filter1 = DE.filter_impacted_stream_side(stream_tail_coordinates[0], perturber_tail_coordinates[0,0])
    filter2 = DE.filter_stream_in_tail_coordinates(stream_tail_coordinates, xlim, ylim, zlim)
    filter3 = np.abs(stream_tail_coordinates[0]) > xmin
    filter = filter1 & filter2 & filter3
    return filter

def convert_instant_to_tail_coordinates(stream_galactic_coordinates: np.ndarray,
                                       host_orbit_galactic_coordinates: np.ndarray,
                                       perturber_galactic_coordinates: np.ndarray,
                                       time_of_interest: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the stream and perturber galactic coordinates to tail coordinates at a given impact time.

    Parameters:
        stream_galactic_coordinates (np.ndarray): Array of stream galactic coordinates.
        host_orbit_galactic_coordinates (np.ndarray): Array of host orbit galactic coordinates.
        perturber_galactic_coordinates (np.ndarray): Array of perturber galactic coordinates.
        impact_time (float): Time of impact.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the stream time coordinate,
        stream tail coordinates, perturber time coordinate, and perturber tail coordinates.
        stream_time_coordinate is the time coordinate of the stream in tail coordinates, i.e. relative to the globular cluster.
        
        the time coordinate is the time ahead of the host globular cluster. 
    """

    # put stream in tail coordinates
    xpT,ypT,zpT,vxT,vyT,vzT,indexes_pT=SOC.transform_from_galactico_centric_to_tail_coordinates(
        *stream_galactic_coordinates,*host_orbit_galactic_coordinates,t0=time_of_interest)
    stream_tail_coordinates = np.array([xpT,ypT,zpT,vxT,vyT,vzT])
    stream_time_coordinate = host_orbit_galactic_coordinates[0,indexes_pT] - time_of_interest
    # get the perturber position at this time in tail coordinates
    x_perT,y_perT,z_perT,vx_perT,vy_perT,vz_perT,indexes_perT=SOC.transform_from_galactico_centric_to_tail_coordinates(\
        *perturber_galactic_coordinates,*host_orbit_galactic_coordinates,t0=time_of_interest)    
    perturber_tail_coordinates = np.array([x_perT,y_perT,z_perT,vx_perT,vy_perT,vz_perT])
    perturber_time_coordinate = host_orbit_galactic_coordinates[0,indexes_perT]
    
    return stream_time_coordinate,stream_tail_coordinates,perturber_time_coordinate,perturber_tail_coordinates
