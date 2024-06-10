import h5py
import os 
import sys 
import numpy as np 
import matplotlib.pyplot as plt
sys.path.append("../code/")
import path_handler as PH #type: ignore
import data_extractors as DE #type: ignore
import parametric_stream_fitting as PSF #type: ignore

# This script is used to plot the moment of impact of the stream with the perturber

def main(perturberName   =   "NGC6584",
         montecarlokey   =   "monte-carlo-010"):
    
    
    outpath=PH.base['manyplots'] + "galacto-impact/"+perturberName+"/"
    outname="moment-of-impact-"+montecarlokey+".png"
    os.makedirs(outpath,exist_ok=True)

    #### FIXED
    GCname          =   "Pal5"
    potential_stream=   "pouliasis2017pii-Pal5-suspects"
    NP              =   int(1e5)
    
    pathStreamOrbit,pathGeometry=get_paths(perturberName,montecarlokey,GCname,potential_stream,NP)
    

    ### Extract geometry stuff 
    b,s_,t_,t,coefficient_time_fit_params,traj,perturb,impact_pos,b_vec_galactic=extract_geometry_data(\
        pathGeometry,montecarlokey)

    #### Extract stream stuff
    streamOrbit=h5py.File(pathStreamOrbit,'r')
    stream_orbit_time_stamps = DE.extract_time_steps_from_stream_orbit(streamOrbit).value
    stream_orbit_index = np.argmin(np.abs(stream_orbit_time_stamps-t))
    stream_galactic_coordinates = DE.get_galactic_coordinates_of_stream(\
            stream_orbit_index,streamOrbit)
    
    # make the curves
    stream_fit,perturber_traj=make_curves(traj,t_,s_,t,coefficient_time_fit_params)
    mytitle="{:s} - {:s} - {:.2f} ~Gyr - {:.2f} pc".format(perturberName,montecarlokey,t,1000*b)
    
    fig,axis=do_plot(stream_fit,perturber_traj,perturb,impact_pos,b_vec_galactic,stream_galactic_coordinates)
    fig.suptitle(mytitle)
    fig.tight_layout()
    fig.savefig(outpath+outname,dpi=300)
    plt.close(fig)
    print("saved", outpath+outname)

    
def do_plot(myfit,perturber_traj,perturb,impact_pos,b_vec_galactic,stream_galactic_coordinates):
    
    plt.style.use('dark_background')
    fig=plt.figure(figsize=(15,2) )
    fig,axis=plt.subplots(1,2,figsize=(10,5),sharey=True)
    axis[0].plot(myfit[0],myfit[1],label='fit')
    axis[0].plot(perturber_traj[0],perturber_traj[1],label='trajectory',linewidth=2)
    axis[0].plot(perturb[0],perturb[1],'o',label='perturber')
    axis[0].scatter(impact_pos[0],impact_pos[1],label='impact')
    axis[0].plot([impact_pos[0],impact_pos[0]+b_vec_galactic[0]],\
        [impact_pos[1],impact_pos[1]+b_vec_galactic[1]],color="w",label='b')
    axis[0].set_xlabel('x [kpc]')
    axis[0].set_ylabel('y [kpc]')

    axis[1].plot(myfit[2],myfit[1],label='fit')
    axis[1].plot(perturber_traj[2],perturber_traj[1],label='trajectory',linewidth=2)
    axis[1].plot(perturb[2],perturb[1],'o',label='perturber')
    axis[1].scatter(impact_pos[2],impact_pos[1],label='impact')
    axis[1].plot([impact_pos[2],impact_pos[2]+b_vec_galactic[2]],\
        [impact_pos[1],impact_pos[1]+b_vec_galactic[1]],color="w",label='b')
    axis[1].set_xlabel('z [kpc]')

    ylims=axis[0].get_ylim()
    xlims=axis[0].get_xlim()
    zlims=axis[1].get_xlim()

    axis[0].scatter(stream_galactic_coordinates[0],stream_galactic_coordinates[1],label='stream',c='w',s=0.1)
    axis[1].scatter(stream_galactic_coordinates[2],stream_galactic_coordinates[1],label='stream',c='w',s=0.1)

    axis[0].set_ylim(ylims)
    axis[0].set_xlim(xlims)
    axis[1].set_xlim(zlims)

    axis[0].legend()
    
    
    return fig,axis
    
    
def make_curves(traj_coeff,t_,s_,t,coefficient_time_fit_params):
    xtraj,ytraj,ztraj=np.polyval(traj_coeff[0],t_),np.polyval(traj_coeff[1],t_),np.polyval(traj_coeff[2],t_)
    stream_fit=PSF.moving_stream_parametric_3D_parabola(s_,t,coefficient_time_fit_params)
    perturber_traj = np.array([xtraj,ytraj,ztraj])
    return stream_fit,perturber_traj

def extract_geometry_data(pathGeometry,montecarlokey):
    with h5py.File(pathGeometry,'r') as geom:
        b  = geom[montecarlokey]['erkal_2015_params']['impact_parameter'][()]
        s_ = np.linspace(*geom[montecarlokey]['parametric_equation_params']['s_range'][:],100)
        t_ = geom[montecarlokey]['parametric_equation_params']['t_time_stamps'][:]
        t = geom[montecarlokey]['parametric_equation_params']['t'][()]
        coefficient_time_fit_params = geom[montecarlokey]['parametric_equation_params']['coefficient_time_fit_params'][()]
        geom[montecarlokey]['parametric_equation_params'].keys()    
        traj=geom[montecarlokey]['parametric_equation_params']['trajectory_coeffs'][:]
        perturb=geom[montecarlokey]['parametric_equation_params']['perturber_position_galactocentric'][:]
        impact_pos=geom[montecarlokey]['parametric_equation_params']['stream_impact_position_galactocentric'][:]
        b_vec_galactic=geom[montecarlokey]['parametric_equation_params']["b_vec_galactic"][:]
    return b,s_,t_,t,coefficient_time_fit_params,traj,perturb,impact_pos,b_vec_galactic

def get_paths(perturberName,montecarlokey,GCname,potential_stream,NP):
    pathStreamOrbit     =   PH.stream_orbit(GCname=GCname,montecarlokey=montecarlokey,potential=potential_stream,NP=NP)
    pathGeometry        =   PH.impact_geometry_results(GCname=GCname,perturber=perturberName,potential=potential_stream)
    return pathStreamOrbit,pathGeometry
    
    
if __name__=="__main__":
    perturberName   =   "NGC6584"
    montecarlokey   =   "monte-carlo-"+str(int(sys.argv[1])).zfill(3)
    main(perturberName,montecarlokey)