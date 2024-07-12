import sys
import os 
sys.path.append('../code/')
import gap_comparative_detections as GCD



if __name__=="__main__":
    config = {
        "GCname":"Pal5",
        "montecarlokey":"monte-carlo-000",
        "NP":int(1e5),
        "internal":"isotropic-plummer",
        "control_potential":"pouliasis2017pii",
        "compare_potential":"pouliasis2017pii-GCNBody",
        "time_of_interest": 0,
        "xlims": [-0.1,0.1],
        "sigmathreshold": 2,
        "method": "Difference",
        "x-coordinate": "tau",
        "x-unit": "s kpc / km",
        "data_output_path": "/scratch2/sferrone/stream_analysis/"}
    
    for i in range(50):
        config["montecarlokey"] = "monte-carlo-" + str(i).zfill(3)
        fulloutname = GCD.make_out_filename(config)
        if os.path.exists(fulloutname):
            print(fulloutname, " already exists, skipping")
            continue 
        else:
            print("Computing for ",fulloutname)
        
        centers,counts_control,counts_compare=GCD.compute_tau_density_profile(config)
        
        GCD.save_density_profles(fulloutname,config,centers,counts_control,counts_compare)