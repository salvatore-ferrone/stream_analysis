import job_functions as JOBS
import sys 

i=sys.argv[1]



targetGCname="Pal5"
montecarlokey="monte-carlo-009"
gfieldname="pouliasis2017pii-GCNBody"
orbitpath="/scratch2/sferrone/simulations/Orbits/"+gfieldname+"/"
perturberlistpath="/obs/sferrone/gc-tidal-loss/inputDATA/PerturberSuspects/Pal5/Pal5-all-perturber-suspects.txt"

fnameNote="-suspects"
# for i in range(50):
montecarlokey="monte-carlo-"+str(i).zfill(3)
print(montecarlokey)
JOBS.ComputeForceOnOrbit(
        targetGCname,
        montecarlokey,
        gfieldname,
        orbitpath,
        perturberlistpath,
        fnameNote=fnameNote)
