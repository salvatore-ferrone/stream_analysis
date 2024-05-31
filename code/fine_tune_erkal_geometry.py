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
## My modules
import parametric_stream_fitting as PSF
import data_extractors as DE 
import StreamOrbitCoords as SOC



#############################################
######## Obtaining the full geometry ########
#############################################
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
   
   
######################################################################
########### Obtaining individual Erkal geometry Parameters ###########
######################################################################
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

######################################################
################ Support computations ################
######################################################
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

