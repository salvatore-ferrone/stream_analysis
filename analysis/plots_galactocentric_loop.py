import sys
import plots_galactocentric_time_of_impact as PGTOI #type: ignore


def plots_loop(perturberName,):
    
    
    for mcarlo_int in range(50):
        montecarlokey="monte-carlo-"+str(mcarlo_int).zfill(3)
        PGTOI.main(perturberName,montecarlokey)
        
        
if __name__=="__main__":
    perturberName=sys.argv[1]
    plots_loop(perturberName)
    print("Done!",perturberName)

        
