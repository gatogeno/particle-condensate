#Load the modules needed for the program
import time
import numpy as np
import sys
from utilities import storer, grid, fields, potentials, evolvers

def main():
    #Read the argument -file name
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-file':
        file=args[1]
    
    #Initialize the counting of time execution
    start_time = time.time()

    #Load parameters
    parameters=storer.reader(file,display=False)

    #Create the grids
    grids=grid.creategrid(parameters)

    #Create the initial configuration of the fields and generate the potentials
    if parameters[4]=='spin0':
        psi=fields.spin0(parameters, grids)
        pot=potentials.potential(parameters, grids, psi,1)
        potnc=potentials.potential(parameters, grids, psi,0)
    if parameters[4]=='spin2':
        psi=fields.spin2(parameters, grids)
        pot=potentials.potential(parameters, grids, psi,1)
        potnc=pot

    #Evolve the system
    if parameters[9][0]=='kickdriftkick' or parameters[9][0]=='driftkickdrift':
        evolvedarray=evolvers.kickdrift(parameters, grids, psi, pot, potnc, 1e-2)
    
    with open(f'{parameters[10]}/timeelapsed.dat', "w") as time_file:
        time_file.write("Time elapsed: %s seconds" % (time.time() - start_time))
    

if __name__ == "__main__":
    main()