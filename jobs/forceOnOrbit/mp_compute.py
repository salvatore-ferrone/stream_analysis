import multiprocessing as mp
import compute 



if __name__=="__main__":
    #dataparams
    GCname = "Pal5"
    MWpotential = "pouliasis2017pii-GCNBody"

    # hyper params
    side_length_factor  =   2
    ntauskip            =   10
    ntskip              =   10
    hyperparams = side_length_factor,ntauskip,ntskip


    montecarloindex=9
    hyperparams = side_length_factor,ntauskip,ntskip
    dataparams = GCname,MWpotential,montecarloindex
    compute.main(dataparams=dataparams,hyperparams=hyperparams)
    # ncpu = mp.cpu_count()
    # NCPU = 15
    # pool = mp.Pool(NCPU)
    # for i in range(50):
    #     montecarlokey = "monte-carlo-{:03d}".format(i)
    #     dataparams = GCname,MWpotential,montecarlokey
    #     pool.apply_async(compute.main,args=(dataparams,hyperparams))
    # pool.close()
    # pool.join()
    # pool.terminate()
    # compute.main(dataparams=dataparams,hyperparams=hyperparams)    