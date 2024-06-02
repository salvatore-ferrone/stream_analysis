import plotly.graph_objects as go
import extract_viewing_geometry as EVG
import h5py
import numpy as np 
import sys
sys.path.append("../code/")
import data_extractors as DE

def main(montecarlo,perturbername,limit=0.200):
    filepath=get_file_path(montecarlo)
    geometryfile=h5py.File(filepath,'r')
    # hard code for the moment, update after bug fix
    perturber_radius=geometryfile[perturbername]["erkal_2015_params"]['plummer_radius'][()]
    
    per_traj,per_pos,stream,impact_pos=EVG.geometry_centered_on_impact(geometryfile,perturbername)
    sphere=make_sphere(perturber_radius,per_pos)
    
    
    traj_trace,stream_trace,b_vec_trace=assemble_geometry_traces(per_traj,per_pos,stream,impact_pos)
    sphere_trace=obtain_sphere_trace(perturbername,sphere)
    x_axis_trace,y_axis_trace,z_axis_trace=get_axis_lines_traces(limit,limit,limit)
    
    fig=go.Figure()
    
    # add the traces
    fig.add_trace(traj_trace)
    fig.add_trace(stream_trace)
    fig.add_trace(b_vec_trace)
    fig.add_trace(sphere_trace)
    # add the axis lines
    fig.add_trace(x_axis_trace)
    fig.add_trace(y_axis_trace)
    fig.add_trace(z_axis_trace)
    fig.update_layout(scene=dict(xaxis=dict(range=[-limit,limit]),
                                 yaxis=dict(range=[-limit,limit]), 
                                 zaxis=dict(range=[-limit,limit]),
                            xaxis_title="X (kpc)", 
                            yaxis_title="Y (kpc)", 
                            zaxis_title="Z (kpc)", 
                            aspectmode='cube'))
    return fig

def obtain_stream_particles(geometry_file,perturber_name):
    t_impact=geometry_file[perturber_name]['parametric_equation_params']['t'][()]
    strea_file=h5py.File(geometry_file.attrs["pathStreamOrbit"], "r")
    my_time_stamps=DE.extract_time_steps_from_stream_orbit(strea_file)
    t_index=np.argmin(np.abs(my_time_stamps.value-t_impact))
    stream=DE.get_galactic_coordinates_of_stream(t_index,strea_file)
    return stream

def obtain_sphere_trace(perturber_name,sphere):
    sphere_trace=go.Surface(x=sphere[0],y=sphere[1],z=sphere[2],
                    surfacecolor=np.zeros_like(sphere[2]), 
                    colorscale=[(0, 'blue'), (1, 'blue')], 
                    showscale=False, 
                    opacity=1,
                    name=perturber_name)
    return sphere_trace

def assemble_geometry_traces(per_traj,per_pos,stream,impact_pos):
    traj_trace=go.Scatter3d(x=per_traj[0],y=per_traj[1],z=per_traj[2],mode='lines',name='Perturber trajectory')
    
    stream_trace=go.Scatter3d(x=stream[0], y=stream[1], z=stream[2], 
                 mode="lines", name="Stream line", line=dict(color="blue", width=1,))
    b_vec_trace=go.Scatter3d(x=[impact_pos[0],per_pos[0]], 
                             y=[impact_pos[1],per_pos[1]], 
                             z=[impact_pos[2],per_pos[2]],
                 mode="lines", name="Impact vector", marker=dict(size=2, color="red"))
    return traj_trace,stream_trace,b_vec_trace

def get_axis_lines_traces(xlim,ylim,zlim):
    x_axis_trace=go.Scatter3d(x=[-xlim,xlim], y=[0,0], z=[0,0], mode="lines", name="X axis", line=dict(color="black", width=0.5,),showlegend=False)
    y_axis_trace=go.Scatter3d(x=[0,0], y=[-ylim,ylim], z=[0,0], mode="lines", name="Y axis", line=dict(color="black", width=0.5,),showlegend=False)
    z_axis_trace=go.Scatter3d(x=[0,0], y=[0,0], z=[-zlim,zlim], mode="lines", name="Z axis", line=dict(color="black", width=0.5,),showlegend=False)
    return x_axis_trace,y_axis_trace,z_axis_trace

def make_sphere(radius,center,npoints=100):
    phi = np.linspace(0, 2*np.pi, npoints)
    theta = np.linspace(0, np.pi, npoints)
    x_sphere = center[0] + radius * np.outer(np.cos(phi), np.sin(theta))
    y_sphere = center[1] + radius * np.outer(np.sin(phi), np.sin(theta))
    z_sphere = center[2] + radius * np.outer(np.ones(np.size(phi)), np.cos(theta))
    return (x_sphere,y_sphere,z_sphere)

def get_file_path(montecarlo,basepath="../impact-geometry-results/"):
    return basepath+"pal5-"+montecarlo+"-erkal-impact-geometry.hdf5"


if __name__ == "__main__":
    montecarlo="monte-carlo-042"
    perturber_name="NGC2808"
    fig=main(montecarlo,perturber_name)
    fig.show()