import multiprocessing as mp
import profile_detections #type: ignore
import copy

if __name__ == "__main__":
    montecarlokey="monte-carlo-010"
    dataparams = {
        "montecarlokey":montecarlokey,
        "GCname":"Pal5",
        "NP":int(1e5),
        "MWpotential0":"pouliasis2017pii",
        "MWpotential1":"pouliasis2017pii-GCNBody",
        "internal_dynamics":"isotropic-plummer"
    }
    ## HYPER PARAMETERS

    hyperparams = {
        "xlims":[-10,10],
        "ylims":[-0.5,0.5],
        "noise_factor":50,
        "sigma":2,
        "box_length":20,
        "do_cross_correlation":True,
        "N_APPlY":2}
    
    
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    n_trials = 50 
    for i in range(n_trials):
        # Create a copy of dataparams for each iteration
        dataparams_copy = copy.deepcopy(dataparams)
        dataparams_copy["montecarlokey"] = "monte-carlo-" + str(i).zfill(3)
        pool.apply_async(profile_detections.main,args=(dataparams_copy,hyperparams))
    pool.close()
    pool.join()

