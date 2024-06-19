import sys
import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift
from utilities import storer, grid, fields, potentials

#Function that removes the comments with # in the initial file
def remove_comments(lines: list[str]) -> list[str]:
    new_lines = []
    for line in lines: 
        if line.startswith("#"):
            continue

        line = line.split(" #")[0]
        if line.strip() != "":
            new_lines.append(line)

    return new_lines

#Function that returns the index in an array for a specific value:
def index_return(array,value):
    return [array.index(i) for i in array if f'{value}' in i][0]

#Function that checks if some parameter is in a list of parameters called container
def checkerlist(param, container):
    try:
        indicestore = next(i for i, item in enumerate(container) if param in item)
    except StopIteration:
        print(f'Parameter {param} is missing. Please enter this parameter in the initial file')
        sys.exit()

    element = container[indicestore].partition("=")[2].replace(" ", "").strip("\n")
    return element
#def checkerlist(param,container):
#    counterexist=0
#    for j in range(0,len(container)):
#        if (param in container[j]):
#            counterexist=counterexist+1
#    if counterexist==0:
#        print(f'Parameter {param} is missing. Please enter this parameter in the initial file')
#        sys.exit()
#    else:
#        indicestore=index_return(container,param)
#        element=container[indicestore].partition("=")[2].replace(" ", "").strip("\n")
#        return element

#Function that generates a progress bar
def progbar(progress,iteration_number):
    size = 60
    status = ""
    progress = progress/iteration_number
    if progress >= 1.:
        progress = 1
    block = int(round(size * progress))
    text="\r{}|{:.0f}%".format("*"*block+""*(size - block),round(progress * 100, 0))
    sys.stdout.write(text)
    sys.stdout.flush()
        
#Functions common to all evolvers
#Function that save in external files the outputs of field, time, energy, mass and potentials
def savingstep(indexstep, parameters, grids, fieldY, pot, potnc, timeline, totalmass, energy):
    if parameters[9][3]==False:
        if indexstep%(parameters[9][2])==0:
            if parameters[5]==1:
                np.save(f'{parameters[10]}/evolution/field_'+str(indexstep)+'.npy', fieldY.field)
            elif parameters[5]==0:
                np.save(f'{parameters[10]}/evolution/fieldpar_'+str(indexstep)+'.npy', fieldY.fieldpar)
            else:
                np.save(f'{parameters[10]}/evolution/field_'+str(indexstep)+'.npy', fieldY.field)
                np.save(f'{parameters[10]}/evolution/fieldpar_'+str(indexstep)+'.npy', fieldY.fieldpar)
    if (indexstep==parameters[9][1]):
        np.save(f'{parameters[10]}/timeline.npy', timeline)
        np.save(f'{parameters[10]}/totalmass.npy', totalmass)
        np.save(f'{parameters[10]}/energy.npy', energy)
        if parameters[5]==1:
            np.save(f'{parameters[10]}/potential.npy', pot)
            np.save(f'{parameters[10]}/field.npy', fieldY.field)
        elif parameters[5]==0:
            np.save(f'{parameters[10]}/potentialpar.npy', potnc)
            np.save(f'{parameters[10]}/fieldpar.npy', fieldY.fieldpar)
        else:
            np.save(f'{parameters[10]}/potential.npy', pot)
            np.save(f'{parameters[10]}/potentialpar.npy', potnc)
            np.save(f'{parameters[10]}/field.npy', fieldY.field)
            np.save(f'{parameters[10]}/fieldpar.npy', fieldY.fieldpar)
        
    return None

#Function that computes kinetic energy
def kineticenergy(parameters, grids, fieldY, selector):
    if parameters[4]=='spin0':
        if selector==1:
            kineticen=0.5*np.sum(np.real(fieldY.field.conjugate()*ifftn(ifftshift(grids[1]*fftshift(fftn(fieldY.field))))))*(grids[0]**parameters[0])
            if kineticen!=0:
                kineticen=kineticen/(np.sum(fieldY.density)*(grids[0]**parameters[0]))
        elif selector==0:
            kineticen=0.5*np.sum(fieldY.fieldpar[1]**2)
            if kineticen!=0:
                kineticen=kineticen/parameters[7][0]
            #    kineticen=kineticen/(np.sum(fieldY.densitypar)*(grids[0]**parameters[0]))
    
    if parameters[4]=='spin2':
        for i in range(parameters[0]):
            for j in range(parameters[0]):
                if i!=j:
                    kineticen=kineticen+0.5*np.sum(np.real(fieldY.field[i][j].conjugate()*ifftn(ifftshift(grids[1]*fftshift(fftn(fieldY.field[i][j]))))))*(grids[0]**parameters[0])/(np.sum(fieldY.density)*(grids[0]**parameters[0]))

    return kineticen

#Function that computes potential energy
def potentialenergy(parameters, grids, fieldY, pot, selector):
    if np.amax(pot)==np.inf:
        potentialen=0
    else:
        if selector==1:
            potentialen=np.sum(pot*fieldY.density)*(grids[0]**parameters[0])
            if potentialen!=0:
                potentialen=potentialen/(np.sum(fieldY.density)*(grids[0]**parameters[0]))
        elif selector==0:
            indices, pesos, poss = fieldY.weightassign
            potentialen=0
            if parameters[7][1]=='nearestgridpoint':
                if parameters[0]==1:
                    potentialen=potentialen+np.sum(pot[indices[:,0]]*pesos)
                if parameters[0]==2:
                    potentialen=potentialen+np.sum(pot[indices[:,0],indices[:,1]]*pesos)
                if parameters[0]==3:
                    potentialen=potentialen+np.sum(pot[indices[:,0],indices[:,1],indices[:,2]]*pesos)
            if parameters[7][1]=='cloudincell':
                for l in range(2**parameters[0]):
                    if parameters[0]==1:
                        potentialen=potentialen+np.sum(pot[indices[l,:,0]]*pesos[l])
                    if parameters[0]==2:
                        potentialen=potentialen+np.sum(pot[indices[l,:,0],indices[l,:,1]]*pesos[l])
                    if parameters[0]==3:
                        potentialen=potentialen+np.sum(pot[indices[l,:,0],indices[l,:,1],indices[l,:,2]]*pesos[l])
            if potentialen!=0:
                potentialen=potentialen/parameters[7][0]
            #potentialen=np.sum(pot*fieldY.densitypar)*(grids[0]**parameters[0])
            #if potentialen!=0:
            #    potentialen=potentialen/(np.sum(fieldY.densitypar)*(grids[0]**parameters[0]))
    
    return potentialen

#Function that corrects the position of a particle if the value goes outside the box according to periodical boundary conditions
def poscorrector(x,grid):
    incr=0.5*(grid[1]-grid[0])
    return (x+np.abs(np.amin(grid)-incr))%(np.amax(grid)+incr+np.abs(np.amin(grid)-incr))-np.abs(np.amin(grid)-incr)
        