import fine_tune_impact_geometry as FTIG 
import sys 

def fine_tune_loop(perturberName):
    GCname          =   "Pal5"
    potential_stream=   "pouliasis2017pii-Pal5-suspects"
    potential_GCs   =   "pouliasis2017pii-GCNBody"
    NP              =   int(1e5)
    
    
    for mcarlo_int in range(50):
        FTIG.main(mcarlo_int, perturberName,
            GCname              =   GCname,\
            potential_stream    =   potential_stream,\
            potential_GCs       =   potential_GCs,\
            NP                  =   NP)
        
    FTIG.combine_temp_files(
        perturberName=perturberName,
        GCname=GCname,
        potential=potential_stream,
)



if __name__=="__main__":
    perturberName=sys.argv[1]
    fine_tune_loop(perturberName=perturberName)
    print("Done!",perturberName)