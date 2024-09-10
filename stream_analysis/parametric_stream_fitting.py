"""
...
Author: Salvatore Ferrone
Date: April 2024 (first functioning full version)
"""



import numpy as np 




#############################################
######## Obtaining the full geometry ########
#############################################
def get_full_impact_geometry_from_parametrization(
    t:float,
    tau:float,
    coefficient_time_fit_params: np.ndarray,
    trajectory_coeffs: np.ndarray,
    time_stamps: np.ndarray,):
    

    rp  =   perturber_position(t,trajectory_coeffs)
    vp  =   perturber_velocity(t,trajectory_coeffs)
    rs  =   moving_parametric_stream(t,tau,coefficient_time_fit_params,time_stamps)
    vs  =   moving_parametric_stream_velocity(t,tau,coefficient_time_fit_params,time_stamps)
    

    impact_parameter = np.linalg.norm(rp-rs)
    bvec    =   get_impact_vector(rs,vs,rp)
    alpha   =   get_imact_alpha_from_b_vector(bvec)
    # the magnitude of b_vec should be the impact impact_parameter
    w_par,w_per=get_parallel_and_perpendicular_velocity_components(vp,vs)

    return impact_parameter,w_par,w_per,alpha
   
   
######################################################################
########### Obtaining individual Erkal geometry Parameters ###########
######################################################################
def get_imact_alpha_from_b_vector(b_vec_tail_basis:np.ndarray):
    """
    Returns the impact parameter in the direction of the position vector.
    The angle is in radians and is the angle up from the the x-axis to the z-axis.
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


def get_impact_vector(rs:np.ndarray,vs:np.ndarray,rp:np.ndarray) -> np.ndarray:
    """
    Returns the impact parameter vector in the refenence frame about the point of impact.
    """
    b=rp-rs
    runit,vunit,Lzunit=get_tail_basis_vectors(rs,vs)
    bvec = np.array([np.dot(b,unit) for unit in [runit,vunit,Lzunit]])
    return bvec


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
    >>> # Given t,tau,coefficient_time_fit_params from the previous functions
    >>> rs = moving_parametric_stream(t,tau,coefficient_time_fit_params)
    >>> vs = moving_parametric_stream_velocity(t,tau,coefficient_time_fit_params)
    >>>  r_unit, v_unit, Lz_unit = get_impact_basis_vectors(rs, vs)
    """
    r_unit = rs / np.linalg.norm(rs)
    v_unit = vs / np.linalg.norm(vs)
    Lz_unit = np.cross(r_unit, v_unit) / np.linalg.norm(np.cross(r_unit, v_unit))
    return r_unit, v_unit, Lz_unit


def get_parallel_and_perpendicular_velocity_components(vs,vp):
    """
    vs: velocity of the stream at impact point, (origin)
    vp: velocity of the perturber at impact point
    """
    assert type(vs)==np.ndarray, "vs must be a numpy array"
    assert type(vp)==np.ndarray, "vp must be a numpy array"
    assert vs.ndim==1, "vs must be a 3D vector"
    assert vp.ndim==1, "vp must be a 3D vector"
    assert vs.shape[0]==3, "vs must be a 3D vector"
    assert vp.shape[0]==3, "vp must be a 3D vector"
    unit_vector = vs/np.linalg.norm(vs)
    dv          = vp-vs
    w_parallel  = np.dot(dv,unit_vector)*unit_vector
    w_perpendicular = dv - w_parallel
    return w_parallel, w_perpendicular

###########################################
########## Finding the impact #############
###########################################
def objective_stream_perturber(fit_params,coeffs_stream,coeffs_perturber,time_stamps):
    t,tau=fit_params
    xs,ys,zs=moving_parametric_stream(t,tau,coeffs_stream,time_stamps)
    xp,yp,zp = perturber_position(t,coeffs_perturber)
    dx,dy,dz=xs-xp,ys-yp,zs-zp
    return np.sqrt(dx**2+dy**2+dz**2)


###########################################
############## THE PERTURBER ##############
###########################################
def perturber_velocity(t,coeffs_perturber):
    xcoeff,ycoeff,zcoefff=separate_coeffs(coeffs_perturber)
    vxcoeff,vycoeff,vzcoeff=np.polyder(xcoeff),np.polyder(ycoeff),np.polyder(zcoefff)
    vx,vy,vz=np.polyval(vxcoeff,t),np.polyval(vycoeff,t),np.polyval(vzcoeff,t)
    return np.array([vx,vy,vz])


def perturber_position(t,coeffs_perturber):
    xcoeff,ycoeff,zcoefff=separate_coeffs(coeffs_perturber)
    xp,yp,zp=np.polyval(xcoeff,t),np.polyval(ycoeff,t),np.polyval(zcoefff,t)
    return np.array([xp,yp,zp])
    

def filter_perturber_orbit(OrbitTuple,tmin,tmax):
    condA=OrbitTuple[0] > tmin
    condB=OrbitTuple[0] < tmax
    cond = np.logical_and(condA,condB)
    t,x,y,z,vx,vy,vz = OrbitTuple
    return np.array([t[cond],x[cond],y[cond],z[cond],vx[cond],vy[cond],vz[cond]])    


##################################################
##### TIME DEPENDENT 3D PARABOLA. Change!! #######
##################################################
def moving_parametric_stream_velocity(t:float,tau:float,coefficients_array:np.ndarray,time_stamps:np.ndarray):
    coeffs=interpolate_coefficients(t,coefficients_array,time_stamps)
    vx,vy,vz = parametric_eq_3D_velocity(coeffs,tau)
    return np.array([vx,vy,vz])


def moving_parametric_stream(t:float,tau:float,coefficients_array:np.ndarray,time_stamps:np.ndarray,):
    assert isinstance(coefficients_array, np.ndarray), "t_dependent_coefficients must be a numpy array"
    coeffs_=interpolate_coefficients(t,coefficients_array,time_stamps)
    x,y,z = parametric_eq_3D(coeffs_,tau)
    return np.array([x,y,z])
    

def interpolate_coefficients(t,coefficients_array,time_stamps):
    assert coefficients_array.shape[0] == len(time_stamps), "The number of time stamps must match the number of coefficients"
    assert coefficients_array.shape[1]%3==0, "The number of coefficients must be divisible by 3, for 3 spatial dimensions"
    coeffs = [np.interp(t,time_stamps,coefficients_array[:,i]) for i in range(coefficients_array.shape[1])]    
    return np.array(coeffs)

##################################################
########## Time independent 3D polynomial ########
##################################################
def objective_3D_parametric_line(coeffs,independent_variable,dependent_variables):
    x,y,z=parametric_eq_3D(coeffs,independent_variable)
    dx=(x-dependent_variables[0,:])
    dy=(y-dependent_variables[1,:])
    dz=(z-dependent_variables[2,:])
    return np.sum(dx**2+dy**2+dz**2)


def initial_guess_3D_parametric_eq(independent_variable,dependent_variables,order=3):
    xcoeff=np.polyfit(independent_variable,dependent_variables[0,:],order)
    ycoeff=np.polyfit(independent_variable,dependent_variables[1,:],order)
    zcoeff=np.polyfit(independent_variable,dependent_variables[2,:],order)
    outcoeff = np.concatenate((xcoeff,ycoeff,zcoeff))
    return outcoeff    


def parametric_eq_3D_velocity(fit_params,independent_variable):
    xcoeff,ycoeff,zcoeff = separate_coeffs(fit_params)
    vxcoeff,vycoeff,vzcoeff=np.polyder(xcoeff),np.polyder(ycoeff),np.polyder(zcoeff)
    vx,vy,vz=np.polyval(vxcoeff,independent_variable),np.polyval(vycoeff,independent_variable),np.polyval(vzcoeff,independent_variable)
    return vx,vy,vz


def parametric_eq_3D(fit_params,independent_variable):
    xcoeff,ycoeff,zcoeff = separate_coeffs(fit_params)
    x,y,z=np.polyval(xcoeff,independent_variable),np.polyval(ycoeff,independent_variable),np.polyval(zcoeff,independent_variable)
    return x,y,z


def separate_coeffs(fit_params):
    len(fit_params)%3==0, "The number of coefficients must be divisible by 3"
    order=len(fit_params)//3 - 1
    xcoeff,ycoeff,zcoeff = fit_params[:order+1],fit_params[order+1:2*order+2],fit_params[2*order+2:]
    return xcoeff,ycoeff,zcoeff



