
import multiprocessing as mp
import parameterize_and_compute_impact as pci  #type: ignore



def loop_over_targets(dataparams,hyperparams):
    GCname,NP,MWpotential,montecarlokey,internal_dynamics = dataparams
    _,n_adjacent_points,nDynTimes,width_factor,streampolyorder,perturberPolyOrder = hyperparams
    NKEEPS = 8
    for i in range(NKEEPS):
        dataparams  =   (GCname,NP,MWpotential,montecarlokey,internal_dynamics)
        hyperparams =   (i,n_adjacent_points,nDynTimes,width_factor,streampolyorder,perturberPolyOrder)
        pci.main(dataparams,hyperparams)

if __name__=="__main__":
    GCname = "Pal5"
    NP = int(1e5)
    MWpotential = "pouliasis2017pii-GCNBody"
    montecarlokey="monte-carlo-000"
    internal_dynamics = "isotropic-plummer"

    # hyper params
    targetnumber        = 3
    n_adjacent_points   = 5
    n_stamps            = 2*n_adjacent_points+1
    nDynTimes           = 2
    width_factor        = 10
    streampolyorder     = 2
    perturberPolyOrder  = 2
    NKEEPS              = 8


    hyperparams =   (0,n_adjacent_points,nDynTimes,width_factor,streampolyorder,perturberPolyOrder)    
    # for i in range(3):
    #     montecarlokey = "monte-carlo-"+str(i).zfill(3)
    #     dataparams  =   (GCname,NP,MWpotential,montecarlokey,internal_dynamics)
    #     loop_over_targets(dataparams,hyperparams)

    ncpu=mp.cpu_count()
    print("Using",ncpu,"cores")
    pool=mp.Pool(ncpu)
    for i in range(50):
        montecarlokey = "monte-carlo-"+str(i).zfill(3)
        dataparams  =   (GCname,NP,MWpotential,montecarlokey,internal_dynamics)
        pool.apply_async(loop_over_targets,args=(dataparams,hyperparams))
    pool.close()
    pool.join()
    print("Done!")