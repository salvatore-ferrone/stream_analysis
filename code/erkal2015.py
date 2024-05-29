"""
following the paper from ERKAL 2015
these are the functions to compute the changes in orbits of particles due to the impact of a GC
"""

import numpy as np 



def cost_function_fixed_G_M_Rs(fit_params:tuple, ycoord:np.ndarray, ydata:np.ndarray, fixed_params:tuple):
    """
    intended to be the input function to scipy.optimize.minimize
    
    In essence, these us the deltaVx, deltaVy, deltaVz functions to compute the change in velocity of the particle that was impacted by the GC
    
    yshift,dVxshift,dVzshift are not presented in the theory for two reasons.
    regarding yshift, the theory choses a coordinate system where it knows the position of the impact along the stream, we however do not know this aprior
    regarding dVxshift and dVzshift, the theory assumes that the particle is on a circular orbit, 
        therefore, any change in these velocities would be due to the impact of the GC, we however do not know the initial velocities of the particles
    

    """
    G,M,Rs=fixed_params
    b, alpha, Wperp, Wpar, y_shift, dVxshift, dVyshift, dVzshift = fit_params    
    y=ycoord-y_shift
    predicted = dV_model_with_offsets(y, G, M, Rs, b, alpha, Wperp, Wpar, y_shift, dVxshift, dVyshift, dVzshift)
    return np.sum((ydata - predicted) ** 2)

def dV_model_with_offsets(y:np.ndarray, G:float, M:float, Rs:float, b:float, alpha:float, Wperp:float, Wpar:float, y_shift:float, dVxshift:float, dVyshift:float, dVzshift:float):
    """
    Calculates the change in velocity vector with offsets.

    Parameters:
    - y (np.ndarray): The input coordinate along the stream.
    - G (float): The gravitational constant.
    - M (float): The mass of the object.
    - Rs (float): The scale radius.
    - b (float): The impact parameter.
    - alpha (float): The angle of the orbit.
    - Wperp (float): The perpendicular velocity dispersion.
    - Wpar (float): The parallel velocity dispersion.
    - y_shift (float): The position of the impact along the stream.
    - dVxshift (float): a change in the x-component of the velocity caused from something other than the impact.
    - dVyshift (float): a change in the y-component of the velocity caused from something other than the impact.
    - dVzshift (float): a change in the z-component of the velocity caused from something other than the impact.

    Returns:
    - numpy.ndarray: The change in velocity vector with offsets.
    """
    ycoord = y - y_shift
    return np.array([
        deltaVx(ycoord, G, M, Rs, b, alpha, Wperp, Wpar) + dVxshift,
        deltaVy(ycoord, G, M, Rs, b, alpha, Wperp, Wpar) + dVyshift,
        deltaVz(ycoord, G, M, Rs, b, alpha, Wperp, Wpar) + dVzshift
    ]).T

def dV_model(y: np.ndarray, G: float, M: float, Rs: float, b: float, alpha: float, Wperp: float, Wpar: float):
    """
    Calculates the change in velocity vector for a given set of parameters.

    Parameters:
    y (array-like): The initial velocity vector.
    G (float): The gravitational constant.
    M (float): The mass of the object.
    Rs (float): The Schwarzschild radius.
    b (float): The impact parameter.
    alpha (float): The angle of the orbit.
    Wperp (float): The perpendicular component of the tidal force.
    Wpar (float): The parallel component of the tidal force.

    Returns:
    array-like: The change in velocity vector.
    """
    return np.array([deltaVx(y, G, M, Rs, b, alpha, Wperp, Wpar),
                     deltaVy(y, G, M, Rs, b, alpha, Wperp, Wpar),
                     deltaVz(y, G, M, Rs, b, alpha, Wperp, Wpar)]).T

def deltaVx(y: np.ndarray, G: float, M: float, Rs: float, b: float, alpha: float, Wperp: float, Wpar: float):
    """
    Calculates the change in velocity in the x-direction (deltaVx) for a given set of parameters.

    Parameters:
    - y: np.ndarray, the coordinate along the stream. 0 is the center position of the impact
    - G: float, the gravitational constant
    - M: float, the mass
    - Rs: float, the Schwarzschild radius
    - b: float, the impact parameter
    - alpha: float, the angle of deflection
    - Wperp: float, the perpendicular velocity component
    - Wpar: float, the parallel velocity component

    Returns:
    - dVx: np.ndarray, the change in velocity in the x-direction
    """
    Wmag=np.sqrt(Wperp**2+Wpar**2)
    term1 = b*(Wmag**2)*np.cos(alpha)
    term2 = y*Wperp*Wpar*np.sin(alpha)
    term3 = (b**2 + Rs**2)*(Wmag**2)
    term4 = (Wperp**2)*(y**2)
    denominator = Wmag*(term3+term4)
    dVx= 2*G*M*(term1+term2)/denominator 
    return dVx

def deltaVy(y:np.ndarray, G:float, M:float, Rs:float, b:float, alpha:float, Wperp:float, Wpar:float):
    """
    Calculates the change in velocity (deltaVy) for a given set of parameters.

    Parameters:
    - y: np.ndarray, the coordinate along the stream. 0 is the center position of the impact
    - G (float): Gravitational constant.
    - M (float): Mass of the object.
    - Rs: Plummer radius of the impactor.
    - b (float): Impact parameter.
    - alpha (float): Angle between the impact parameter and the perpendicular component of the velocity.
    - Wperp (float): Perpendicular component of the velocity.
    - Wpar (float): Parallel component of the velocity.

    Returns:
    - dVy: np.ndarray, the change in velocity in the

    """
    Wmag = np.sqrt(Wperp**2 + Wpar**2)
    term3 = (b**2 + Rs**2) * (Wmag**2)
    term4 = (Wperp**2) * (y**2)
    denominator = Wmag * (term3 + term4)
    dVy = -2 * G * (M * (Wperp**2) * y) / denominator
    return dVy

def deltaVz(y:np.ndarray, G:float, M:float, Rs:float, b:float, alpha:float, Wperp:float, Wpar:float):
    """
    Calculates the change in velocity in the z-direction (deltaVz) for a given set of parameters.

    Parameters:
    - y: np.ndarray, the coordinate along the stream. 0 is the center position of the impact
    - G: Gravitational constant.
    - M: Mass of the gravitational source.
    - Rs: Plummer radius of the impactor.
    - b: Impact parameter.
    - alpha: Angle between the impact parameter vector and the z-axis.
    - Wperp: Perpendicular component of the velocity vector.
    - Wpar: Parallel component of the velocity vector.

    Returns:
    - dVz: Change in velocity in the z-direction.

    """
    Wmag = np.sqrt(Wperp**2 + Wpar**2)
    term1 = (b * Wmag**2) * np.sin(alpha)
    term2 = -y * Wperp * Wpar * np.cos(alpha)
    term3 = (b**2 + Rs**2) * (Wmag**2)
    term4 = (Wperp**2) * (y**2)
    denominator = Wmag * (term3 + term4)
    dVz = 2 * G * M * (term1 + term2) / denominator 
    return dVz

def delta_y_critical_points(G:float,M:float,Wperp:float,Wpar:float,b:float,Rs:float):
    """
    PURPOSE:
        The critical points of delta Y as shown in Fig 3
        
    ARGUMENTS:
        G: gravitational constant
        M: mass of the GC perturber
        b: impact parameter
        Wperp: component of perturber velocity perpendicular to orbit plane
        Wpar: component of perturber velocity parralel to orbit plane
        Rs: plummer radius of the perturber
        
    returns:
        array of critical points
        dVmax = maximum change in velocity
        y_max = position of maximum change in velocity
        yfwhm = full width at half maximum of the change in velocity, like sigma
    """
    
    pythag = np.sqrt(b**2 + Rs**2)
    mu = G*M
    W= np.sqrt(Wperp**2+Wpar**2)
    dVmax= mu * Wperp / (W**2 * pythag)
    y_max = W*pythag/Wperp
    yfwhm = 3.5 * W*pythag/Wperp
    
    return np.array([dVmax,y_max,yfwhm])
