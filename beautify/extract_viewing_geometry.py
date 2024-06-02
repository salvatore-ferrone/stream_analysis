"""
This module will store our interactive graphs 
"""

import numpy as np
import plotly.graph_objects as go
import sys 
sys.path.append("../code/")
import parametric_stream_fitting as PSF


def geometry_centered_on_impact(geometry_file,perturber):
    
    per_traj,per_pos,stream,impact_pos=geometry_galactocentric(geometry_file,perturber)
    
    x_per_traj,y_per_traj,z_per_traj=per_traj[0] - impact_pos[0],per_traj[1] - impact_pos[1],per_traj[2] - impact_pos[2]
    x_per,y_per,z_per=per_pos[0] - impact_pos[0],per_pos[1] - impact_pos[1],per_pos[2] - impact_pos[2]
    xs,ys,zs = stream[0] - impact_pos[0],stream[1] - impact_pos[1],stream[2] - impact_pos[2]
    xs_,ys_,zs_=impact_pos[0]-impact_pos[0],impact_pos[1]-impact_pos[1],impact_pos[2]-impact_pos[2]
    
    per_traj=[x_per_traj,y_per_traj,z_per_traj]
    per_pos=[x_per,y_per,z_per]
    stream=[xs,ys,zs]
    impact_pos=[xs_,ys_,zs_]
    return per_traj,per_pos,stream,impact_pos


def center_stream_particles_on_impact(geometry_file,perturber,stream):
    _,_,_,impact_pos=geometry_galactocentric(geometry_file,perturber)
    xpart,ypart,zpart=stream[0] - impact_pos[0],stream[1] - impact_pos[1],stream[2] - impact_pos[2]
    return xpart,ypart,zpart

def geometry_galactocentric(geometry_file,perturber):
    x_per_traj,y_per_traj,z_per_traj    = perturber_trajectory(geometry_file, perturber)
    x_per,y_per,z_per                   = perturber_position_at_impact(geometry_file, perturber)
    xs,ys,zs                            = stream_line(geometry_file, perturber)
    xs_,ys_,zs_                         = stream_line_impact_position(geometry_file, perturber,)
    
    per_traj=[x_per_traj,y_per_traj,z_per_traj]
    per_pos=[x_per,y_per,z_per]
    stream=[xs,ys,zs]
    impact_pos=[xs_,ys_,zs_]
    return per_traj,per_pos,stream,impact_pos


def stream_line_impact_position(geometry_file,perturber,):
    s=geometry_file[perturber]['parametric_equation_params']["s"][()]
    t_impact=geometry_file[perturber]['parametric_equation_params']["t"][()]
    time_stream_params=geometry_file[perturber]['parametric_equation_params']['coefficient_time_fit_params'][:]
    xs_,ys_,zs_=PSF.moving_stream_parametric_3D_parabola(s,t_impact,time_stream_params)
    return xs_,ys_,zs_


def stream_line(geometry_file,perturber,n_sampling_points=500):
    s_range=np.linspace(*geometry_file[perturber]['parametric_equation_params']["s_range"][:],n_sampling_points)
    t_impact=geometry_file[perturber]['parametric_equation_params']["t"][()]
    time_stream_params=geometry_file[perturber]['parametric_equation_params']['coefficient_time_fit_params'][:]
    xs,ys,zs=PSF.moving_stream_parametric_3D_parabola(s_range,t_impact,time_stream_params)
    return xs,ys,zs


def perturber_position_at_impact(geometry_file,perturber):
    t_impact=geometry_file[perturber]['parametric_equation_params']["t"][()]
    x_per=np.polyval(geometry_file[perturber]['parametric_equation_params']["trajectory_coeffs"][0],t_impact)
    y_per=np.polyval(geometry_file[perturber]['parametric_equation_params']["trajectory_coeffs"][1],t_impact)
    z_per=np.polyval(geometry_file[perturber]['parametric_equation_params']["trajectory_coeffs"][2],t_impact)
    return x_per,y_per,z_per
  
  
def perturber_trajectory(geometry_file,perturber):
    t_time_stamps=geometry_file[perturber]['parametric_equation_params']["t_time_stamps"][:]
    x_per_traj=np.polyval(geometry_file[perturber]['parametric_equation_params']["trajectory_coeffs"][0],t_time_stamps)
    y_per_traj=np.polyval(geometry_file[perturber]['parametric_equation_params']["trajectory_coeffs"][1],t_time_stamps)
    z_per_traj=np.polyval(geometry_file[perturber]['parametric_equation_params']["trajectory_coeffs"][2],t_time_stamps)
    return x_per_traj,y_per_traj,z_per_traj
    

