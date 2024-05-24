import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift

def potential(parameters, grids, fieldY, selector):
    #selector is a parameter to be used for the full potential. It is 1 if the potential will be for the condensate and 0 if it will be for the particles. This since they are different in the self-interaction part.
    
    #testing: potential well
    if parameters[8][0]=='testing':
        if parameters[0]==1:
            pot=np.zeros((parameters[2]))
            for i in range(parameters[2]):
                if grids[2][i]<0 or grids[2][i]>parameters[8][1]:
                    pot[i]=np.inf
                else:
                    pot[i]=0
        if parameters[0]==2:
            pot=np.zeros((parameters[2],parameters[2]))
            for i in range(parameters[2]):
                for j in range(parameters[2]):
                    if (grids[2][i][j]<0 or grids[2][i][j]>parameters[8][1]) or (grids[4][i][j]<0 or grids[4][i][j]>parameters[8][1]):
                        pot[i][j]=np.inf
                    else:
                        pot[i][j]=0
        if parameters[0]==3:
            pot=np.zeros((parameters[2],parameters[2],parameters[2]))
            for i in range(parameters[2]):
                for j in range(parameters[2]):
                    for k in range(parameters[2]):
                        if (grids[2][i][j][k]<0 or grids[2][i][j][k]>parameters[8][1]) or (grids[4][i][j][k]<0 or grids[4][i][j][k]>parameters[8][1]) or (grids[6][i][j][k]<0 or grids[6][i][j][k]>parameters[8][1]):
                            pot[i][j][k]=np.inf
                        else:
                            pot[i][j][k]=0
                            
    #full
    if parameters[8][0]=='full':
        if parameters[0]==1:
            pot=np.zeros((parameters[2]))
            gammad=2 #Factor that goes in the gravitational part
            #harmonical part
            pot=pot+0.5*(parameters[8][2]**2)*grids[2]**2
        if parameters[0]==2:
            pot=np.zeros((parameters[2],parameters[2]))
            gammad=2*np.pi #Factor that goes in the gravitational part
            #harmonical part
            pot=pot+0.5*(parameters[8][2]**2)*(grids[2]**2+grids[4]**2)
        if parameters[0]==3:
            pot=np.zeros((parameters[2],parameters[2],parameters[2]))
            gammad=4*np.pi #Factor that goes in the gravitational part
            #harmonical part
            pot=pot+0.5*(parameters[8][2]**2)*(grids[2]**2+grids[4]**2+grids[6]**2)
        
        if parameters[5]==1:
            #gravitational part
            if parameters[8][1]==True:
                potk=-(gammad*fftshift(fftn(fieldY.density-(parameters[3]/(parameters[1]**parameters[0])))))/(grids[1]  + (grids[1]==0))
                pot=pot+np.real(ifftn(ifftshift(potk)))
            #self-interacting part
            if selector==1:
                pot=pot+parameters[8][3]*fieldY.density
            if selector==0:
                pot=pot*0
            
        elif parameters[5]==0:
            #gravitational part
            if parameters[8][1]==True:
                potk=-(gammad*fftshift(fftn(fieldY.densitypar-(parameters[3]/(parameters[1]**parameters[0])))))/(grids[1]  + (grids[1]==0))
                pot=pot+np.real(ifftn(ifftshift(potk)))
            #self-interacting part
            if selector==1:
                pot=pot*0
            if selector==0:
                pot=pot+parameters[8][3]*2.0*fieldY.densitypar
            
        else:
            #gravitational part
            if parameters[8][1]==True:
                potk=-(gammad*fftshift(fftn(fieldY.density+fieldY.densitypar-(parameters[3]/(parameters[1]**parameters[0])))))/(grids[1]  + (grids[1]==0))
                pot=pot+np.real(ifftn(ifftshift(potk)))
            #self-interacting part
            if selector==1:
                pot=pot+parameters[8][3]*(fieldY.density+2.0*fieldY.densitypar)
            if selector==0:
                pot=pot+parameters[8][3]*2.0*(fieldY.density+fieldY.densitypar)
        
    return pot
