import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift
from scipy.interpolate import interp1d, RegularGridInterpolator
from utilities import potentials
from utilities import additionalfunctions as adf

#Function that computes the acceleration for particles
def acceleration(parameters, grids, fieldY, potnc):
    dU=np.gradient(potnc,grids[0],edge_order=2)
    indices, pesos, poss = fieldY.weightassign
    acel=np.zeros((len(fieldY.fieldpar[0]),parameters[0]))
    
    if parameters[7][1]=='nearestgridpoint':
        if parameters[0]==1:
            acel[:,0]=-dU[indices[:,0]]*pesos
        if parameters[0]==2:
            acel[:,0]=-dU[0][indices[:,0],indices[:,1]]*pesos
            acel[:,1]=-dU[1][indices[:,0],indices[:,1]]*pesos
        if parameters[0]==3:
            acel[:,0]=-dU[0][indices[:,0],indices[:,1],indices[:,2]]*pesos
            acel[:,1]=-dU[1][indices[:,0],indices[:,1],indices[:,2]]*pesos
            acel[:,2]=-dU[2][indices[:,0],indices[:,1],indices[:,2]]*pesos
    
    if parameters[7][1]=='cloudincell':
        if parameters[0]==1:
            for l in range(2**parameters[0]):
                acel[:,0]=acel[:,0]-dU[indices[l,:,0]]*pesos[l]
        if parameters[0]==2:
            for l in range(2**parameters[0]):
                acel[:,0]=acel[:,0]-dU[0][indices[l,:,0],indices[l,:,1]]*pesos[l]
                acel[:,1]=acel[:,1]-dU[1][indices[l,:,0],indices[l,:,1]]*pesos[l]
        if parameters[0]==3:
            for l in range(2**parameters[0]):
                acel[:,0]=acel[:,0]-dU[0][indices[l,:,0],indices[l,:,1],indices[l,:,2]]*pesos[l]
                acel[:,1]=acel[:,1]-dU[1][indices[l,:,0],indices[l,:,1],indices[l,:,2]]*pesos[l]
                acel[:,2]=acel[:,2]-dU[2][indices[l,:,0],indices[l,:,1],indices[l,:,2]]*pesos[l]

    return acel

#Function that updates the positions in the kickdrift evolver
def drift(coef, chi, dt, parameters, grids, fieldY, pot, potnc):
    #coef is the factor 0.5 if we have done drift-kick-drift or 1 if it is kick-drift-kick 
    if parameters[5]==1:
        fieldYk=fftshift(fftn(fieldY.field))
        fieldYk=np.exp(-chi*coef*0.5*grids[1]*dt)*fieldYk
        fieldY.field=ifftn(ifftshift(fieldYk))
        #updating the potential
        pot=potentials.potential(parameters, grids, fieldY, 1)
        
    elif parameters[5]==0:
        fieldY.fieldpar[0]=fieldY.fieldpar[0]+fieldY.fieldpar[1]*coef*dt
        #correct the position according to periodical boundary conditions
        fieldY.fieldpar[0]=adf.poscorrector(fieldY.fieldpar[0],fieldY.getunigrid())
        #updating the potential
        potnc=potentials.potential(parameters, grids, fieldY, 0)
        
    else:
        fieldY.fieldpar[0]=fieldY.fieldpar[0]+fieldY.fieldpar[1]*coef*dt
        #correct the position according to periodical boundary conditions
        fieldY.fieldpar[0]=adf.poscorrector(fieldY.fieldpar[0],fieldY.getunigrid())
        fieldYk=fftshift(fftn(fieldY.field))
        fieldYk=np.exp(-chi*coef*0.5*grids[1]*dt)*fieldYk
        fieldY.field=ifftn(ifftshift(fieldYk))  
        #updating the potential
        pot=potentials.potential(parameters, grids, fieldY, 1)
        potnc=potentials.potential(parameters, grids, fieldY, 0)
  
    return None

#Function that updates the velocities in the kickdrift evolver
def kick(coef, chi, dt, parameters, grids, fieldY, pot, potnc):
    #coef is the factor 1 if we have done drift-kick-drift or 0.5 if it is kick-drift-kick 
    if parameters[5]==1:
        fieldY.field=np.exp(-chi*coef*dt*pot)*fieldY.field
    elif parameters[5]==0:
        acel=acceleration(parameters, grids, fieldY, potnc)
        fieldY.fieldpar[1]=fieldY.fieldpar[1]+acel*coef*dt
    else:
        acel=acceleration(parameters, grids, fieldY, potnc)
        fieldY.fieldpar[1]=fieldY.fieldpar[1]+acel*coef*dt
        fieldY.field=np.exp(-chi*coef*dt*pot)*fieldY.field

    return None

def kickdrift(parameters, grids, fieldY, pot, potnc, factordt):
    #define elements to store physical quantities during the evolution
    if parameters[4]=='spin0':
        energy=np.zeros((3,3,parameters[9][1]))
        totalmass=np.zeros((3,parameters[9][1]))
    else:
        energy=np.zeros((3,parameters[9][1]))
        totalmass=np.zeros(parameters[9][1])
    timeline=np.zeros(parameters[9][1])
    
    #Choose the 1 or i according if it is imaginary time propagation or not
    if parameters[9][3]==True:
        chi=1.0
    if parameters[9][3]==False:
        chi=1.0j
     
    t=0 #set initial time
    #The process of evolution
    for h in range(parameters[9][1]):
        #We adapt the timesteps
        maxpot=np.maximum(np.amax(np.abs(pot)),np.amax(np.abs(potnc)))
        if 1/maxpot==0 or maxpot==0:
            dt=factordt*(grids[0]**2)/6.0
        else:
            dt=factordt*np.minimum((grids[0]**2)/6.0,1/maxpot)
        
        if parameters[9][0]=='driftkickdrift': 
            drift(0.5, chi, dt, parameters, grids, fieldY, pot, potnc)
            kick(1, chi, dt, parameters, grids, fieldY, pot, potnc)
            drift(0.5, chi, dt, parameters, grids, fieldY, pot, potnc)
        if parameters[9][0]=='kickdriftkick':
            kick(0.5, chi, dt, parameters, grids, fieldY, pot, potnc)
            drift(1, chi, dt, parameters, grids, fieldY, pot, potnc)
            kick(0.5, chi, dt, parameters, grids, fieldY, pot, potnc)

        #Normalization if we use imaginary time
        if parameters[9][3]==True:
            if parameters[5]==1:
                integnorm=np.sum(fieldY.density)*(grids[0]**parameters[0])
                fieldY.field=fieldY.field*np.sqrt(1/integnorm)
        
        #Store the energy, number and time
        if parameters[4]=='spin0':
            #Kinetic energy particle and condensate
            energy[0][1][h]=adf.kineticenergy(parameters, grids, fieldY, 0)
            energy[1][1][h]=adf.kineticenergy(parameters, grids, fieldY, 1)
            energy[2][1][h]=energy[0][1][h]+energy[1][1][h]
            #Potential energy particle and condensate
            energy[0][2][h]=adf.potentialenergy(parameters, grids, fieldY, potnc, 0)
            energy[1][2][h]=adf.potentialenergy(parameters, grids, fieldY, pot, 1)
            energy[2][2][h]=energy[0][2][h]+energy[1][2][h]
            #Total energy particle and condensate
            energy[0][0][h]=energy[0][1][h]+energy[0][2][h]
            energy[1][0][h]=energy[1][1][h]+energy[1][2][h]
            energy[2][0][h]=energy[0][0][h]+energy[1][0][h]
            #Total mass particle and condensate
            totalmass[0][h]=np.sum(fieldY.densitypar)*(grids[0]**parameters[0])
            totalmass[1][h]=np.sum(fieldY.density)*(grids[0]**parameters[0])
            totalmass[2][h]=totalmass[0][h]+totalmass[1][h]
        else:
            energy[1][h]=adf.kineticenergy(parameters, grids, fieldY,1)
            energy[2][h]=adf.potentialenergy(parameters, grids, fieldY, pot, 1)
            energy[0][h]=energy[1][h]+energy[2][h]
            totalmass[h]=np.sum(fieldY.density)*(grids[0]**parameters[0])
        timeline[h]=t
        
        t=t+dt
        
        #Saving the data
        adf.savingstep(h, parameters, grids, fieldY, pot, potnc, timeline, totalmass, energy)        
        adf.progbar(h+1,parameters[9][1])
    
    return energy, totalmass, timeline
    
    
    
        
        
    
   