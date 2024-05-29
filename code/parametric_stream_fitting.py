import numpy as np 
import sys 
codefilepath = "/obs/sferrone/gc-tidal-loss/forceOnOrbit/"
sys.path.append(codefilepath)




def objective_distance_perturber_and_moving_stream(
    fit_params:tuple,
    t_dependent_stream_coeffs:np.ndarray,
    pertruber_parametric_coeffs:np.ndarray):
    """
    Calculate the objective distance between a moving stream and a perturber.

    Provided that the stream and the perturber are both parametric, this function calculates the distance between the two objects.
    
    
    Parameters:
    fit_params (tuple): A tuple containing two float values representing the parameters of the fit.
    t_dependent_stream_coeffs (np.ndarray): A numpy array of shape (9,) containing the coefficients for the t-dependent stream.
    pertruber_parametric_coeffs (np.ndarray): A numpy array of shape (3,) containing the coefficients for the perturber.

    Returns:
    float: The objective distance between the moving stream and the perturber.
    
    example:
    >>> # s is the stream coordinate, t is the simulation time. Both are in integration units
    >>> s=1.0 # time ahead of the COM
    >>> t=2.0 # the simulation time (in integration units)
    >>> # the terms of the polynomial for the 3D line are given by
    >>> a0,a1,a2 = 1.0, 2.0, 3.0
    >>> b0,b1,b2 = 4.0, 5.0, 6.0
    >>> c0,c1,c2 = 7.0, 8.0, 9.0
    >>> # the coefficients for the perturber's path
    >>> alpha0,alpha1,alpha2 = 1.0, 2.0, 3.0
    >>> fit_params = np.array([s,t])
    >>> t_dependent_stream_coeffs = np.array([a0,a1,a2,b0,b1,b2,c0,c1,c2])
    >>> pertruber_parametric_coeffs = np.array([alpha0,alpha1,alpha2])
    >>> # compute the distance of this stream coordiante to the perturber
    >>> impact_parameter = objective_distance_perturber_and_moving_stream(fit_params, t_dependent_stream_coeffs, pertruber_parametric_coeffs)
    """
    
    assert len(fit_params) == 2, "fit_params must have 2 elements"
    s,t=fit_params
    assert isinstance(s, float), "s must be a float"
    assert isinstance(t, float), "t must be a float"
    assert isinstance(t_dependent_stream_coeffs, np.ndarray), "t_dependent_coefficients must be a numpy array"
    assert isinstance(t_dependent_stream_coeffs, np.ndarray), "t_dependent_coefficients must be a numpy array"
    assert t_dependent_stream_coeffs.shape[0] == 9, "t_dependent_coefficients must have 9 elements"
    assert pertruber_parametric_coeffs.shape[0] == 3, "t_dependent_coefficients must have 3 elements"
    
    position_x_perturber = np.polyval(pertruber_parametric_coeffs[0], t)
    position_y_perturber = np.polyval(pertruber_parametric_coeffs[1], t)
    position_z_perturber = np.polyval(pertruber_parametric_coeffs[2], t)
    position_perturber=np.array([position_x_perturber, position_y_perturber, position_z_perturber])
    position_stream = moving_stream_parametric_3D_parabola(s, t, t_dependent_stream_coeffs)
    
    objective = np.linalg.norm(position_stream - position_perturber)
    
    return objective


def velocity_moving_stream_parametric_3D_parabola(\
    s:float,
    t:float,
    t_dependent_coefficients:np.ndarray,):
    """
    Compute the moving stream velocity using the given s, t, and t_dependent_coefficients.
    
    dv/dt = ∂v/∂t + ∂v/∂s * (ds/dt)
    
    ds/dt = 1 
    
    NOTE ds/dt = 1 by definition, since s is the stream coordinate measured in time. 
        for instance, 1 unit of s is where the globular cluster will be in 1 unit of time from now. 
            Therefore, the derivative of s with respect to time is always 1. 
            if I choose s to be a spatial coordinate, then ds/dt would be the velocity of the globular cluster along the path.
    
    Parameters:
    s : float. Intended to be the stream coordinate, distance from the COM measured in time. positive is ahead, negative is behind
    t : float. Intended to be the time coordinate, the simulation time
    t_dependent_coefficients : numpy.ndarray. The coefficients of the moving stream. How the shape and position of the stream changes in time. 
            the first element in the array is the linear term, the second element is the constant term
    
    returns:
    numpy.ndarray. The x, y, and z coordinates of a point along the moving stream
    """


    assert isinstance(t_dependent_coefficients, np.ndarray), "t_dependent_coefficients must be a numpy array"
    assert t_dependent_coefficients.shape[0] == 9, "t_dependent_coefficients must have 9 elements"

    assert t_dependent_coefficients.shape[1] == 2, "t_dependent_coefficients must be linear (2 dimensions per coefficient)"
    
    a0_m=t_dependent_coefficients[0,0]
    a1_m=t_dependent_coefficients[1,0]
    a2_m=t_dependent_coefficients[2,0]
    b0_m=t_dependent_coefficients[3,0]
    b1_m=t_dependent_coefficients[4,0]
    b2_m=t_dependent_coefficients[5,0]
    c0_m=t_dependent_coefficients[6,0]
    c1_m=t_dependent_coefficients[7,0]
    c2_m=t_dependent_coefficients[8,0]

    dvxdt,dvxds = (a0_m+a1_m*s+a2_m*s**2) , (np.polyval(t_dependent_coefficients[1],t) + 2*s*np.polyval(t_dependent_coefficients[2],t))
    dvydt,dvyds = (b0_m+b1_m*s+b2_m*s**2) , (np.polyval(t_dependent_coefficients[4],t) + 2*s*np.polyval(t_dependent_coefficients[5],t))
    dvzdt,dvzds = (c0_m+c1_m*s+c2_m*s**2) , (np.polyval(t_dependent_coefficients[7],t) + 2*s*np.polyval(t_dependent_coefficients[8],t))
    vx=dvxdt+dvxds
    vy=dvydt+dvyds
    vz=dvzdt+dvzds
    
    return np.array([vx,vy,vz])


def moving_stream_parametric_3D_parabola(\
    s:float,
    t:float,
    t_dependent_coefficients:np.ndarray,):
    """
    Compute the moving stream using the given s, t, and t_dependent_coefficients.
    
    Parameters:
    s : float. Intended to be the stream coordinate, distance from the COM measured in time. positive is ahead, negative is behind
    t : float. Intended to be the time coordinate, the simulation time
    t_dependent_coefficients : numpy.ndarray. The coefficients of the moving stream. How the shape and position of the stream changes in time. 
    
    returns:
    numpy.ndarray. The x, y, and z coordinates of a point along the moving stream
    """


    assert isinstance(t_dependent_coefficients, np.ndarray), "t_dependent_coefficients must be a numpy array"
    assert t_dependent_coefficients.shape[0] == 9, "t_dependent_coefficients must have 9 elements"
    
    a0=np.polyval(t_dependent_coefficients[0],t)
    a1=np.polyval(t_dependent_coefficients[1],t)
    a2=np.polyval(t_dependent_coefficients[2],t)
    b0=np.polyval(t_dependent_coefficients[3],t)
    b1=np.polyval(t_dependent_coefficients[4],t)
    b2=np.polyval(t_dependent_coefficients[5],t)
    c0=np.polyval(t_dependent_coefficients[6],t)
    c1=np.polyval(t_dependent_coefficients[7],t)
    c2=np.polyval(t_dependent_coefficients[8],t)
    
    x=a0+a1*s+a2*s**2
    y=b0+b1*s+b2*s**2
    z=c0+c1*s+c2*s**2
    return np.array([x,y,z])
    

def constrain_3D_parabola_initial_guess(\
    independent_variable:np.ndarray, \
    dependent_variable:np.ndarray):
    
    """
    Constrain the initial guess by fitting the data to each dimension independently.
    
    Parameters:
    -----------
    independent_variable : numpy.ndarray
        The independent variable data.
        
    dependent_variable : numpy.ndarray
        The dependent variable data.
        
    Returns:
    --------
    numpy.ndarray
        The constrained initial guess parameters.
    """
    
    assert isinstance(independent_variable, np.ndarray), "independent_variable must be a numpy array"
    assert len(independent_variable.shape) == 1, "independent_variable must be a one dimensional numpy array"
    assert isinstance(dependent_variable, np.ndarray), "dependent_variable must be a numpy array"
    assert len(dependent_variable.shape) == 2, "dependent_variable must be a two dimensional numpy array"
    assert dependent_variable.shape[0] == 3, "dependent_variable must have 3 rows"
    
    ax0,vx0,x0=np.polyfit(independent_variable,dependent_variable[0,:],2)
    ay0,vy0,y0=np.polyfit(independent_variable,dependent_variable[1,:],2)
    az0,vz0,z0=np.polyfit(independent_variable,dependent_variable[2,:],2)
    
    initial_guess =[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]
    
    return initial_guess
    

def objective_parametric_3D_parabola(fit_params, independent_variable, dependent_variable):
    """
    Calculate the objective function value for fitting a parametric 3D parabola.

    Parameters:
    ----------
    fit_params : array_like
        The parameters of the parametric 3D parabola.
    independent_variable : array_like
        The independent variable values.
    dependent_variable : array_like
        The dependent variable values.

    Returns:
    -------
    float
        The objective function value.

    Notes:
    ------
    The objective function value is calculated as the sum of squared differences between the calculated
    values of the parametric 3D parabola and the dependent variable values.

    """
    assert len(fit_params) == 9, "fit_params must have 9 elements"
    assert isinstance(independent_variable, np.ndarray), "independent_variable must be a numpy array"
    assert len(independent_variable.shape) == 1, "independent_variable must be a one dimensional numpy array"
    assert isinstance(dependent_variable, np.ndarray), "dependent_variable must be a numpy array"
    assert len(dependent_variable.shape) == 2, "dependent_variable must be a two dimensional numpy array"
    assert dependent_variable.shape[0] == 3, "dependent_variable must have 3 rows"
    assert dependent_variable.shape[1] == independent_variable.shape[0], "dependent_variable must have the same number of columns as independent_variable"
    
    x, y, z = parametric_3D_parabola(fit_params, independent_variable)
    dx = x - dependent_variable[0]
    dy = y - dependent_variable[1]
    dz = z - dependent_variable[2]
    return np.sum(dx**2 + dy**2 + dz**2)


def parametric_3D_parabola(\
    fit_params:np.ndarray, \
    independent_variable:np.ndarray):
    """
    Compute the 3D parabola using the given fit parameters and independent variable.

    Parameters
    ----------
    fit_params : array_like
        The fit parameters for the 3D parabola. It must have 9 elements in the following order:
        [a0, a1, a2, b0, b1, b2, c0, c1, c2], where a, b, and c are coefficients of the parabola.
    independent_variable : array_like
        The independent variable values used to compute the parabola. It must be a one-dimensional numpy array.

    Returns
    -------
    numpy.ndarray
        An array containing the computed x, y, and z values of the 3D parabola.

    Raises
    ------
    AssertionError
        If the length of fit_params is not 9 or if independent_variable is not a one-dimensional numpy array.

    Examples
    --------
    >>> fit_params = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> independent_variable = np.array([1, 2, 3, 4, 5])
    >>> parametric_3D_parabola(fit_params, independent_variable)
        array([ [  6,  17,  34,  57,  86],
                [ 15,  38,  73, 120, 179],
                [ 24,  59, 112, 183, 272]])

    """
    
    assert len(fit_params) == 9, "fit_params must have 9 elements"
    assert isinstance(independent_variable, np.ndarray), "independent_variable must be a numpy array"
    assert len(independent_variable.shape) == 1, "independent_variable must be a one dimensional numpy array"
    
    a = fit_params[0:3]
    b = fit_params[3:6]
    c = fit_params[6:9]
    
    x = a[0] + a[1] * independent_variable + a[2] * independent_variable**2
    y = b[0] + b[1] * independent_variable + b[2] * independent_variable**2
    z = c[0] + c[1] * independent_variable + c[2] * independent_variable**2
    
    return np.array([x, y, z])

