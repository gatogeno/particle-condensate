import numpy as np
from utilities import additionalfunctions as adf

#spin0 boson
class spin0:
    def __init__(self, parameters, grids):
        #Define the particle array for position and velocities
        self.fieldpar=np.array([np.zeros((parameters[7][0],parameters[0])),np.zeros((parameters[7][0],parameters[0]))])
        
        #Assign the values for particles according the initial conditions
        if parameters[6][1][0]!='None':
            if parameters[6][1][0]=='testing':
                for i in range(parameters[7][0]):
                    for j in range(parameters[0]):
                        self.fieldpar[0][i][j]=np.random.uniform(-parameters[1]/4.0,parameters[1]/4.0)
                        self.fieldpar[1][i][j]=np.sign(np.random.uniform(-1,1))*0
            if parameters[6][1][0]=='random':
                for i in range(parameters[7][0]):
                    module=np.random.uniform(0,1)
                    if parameters[0]==1:
                        self.fieldpar[0][i][0]=np.random.uniform(np.amin(grids[2]),np.amax(grids[2]))
                        self.fieldpar[1][i][0]=module
                    if parameters[0]==2:
                        self.fieldpar[0][i][0]=np.random.uniform(np.amin(grids[2]),np.amax(grids[2]))
                        self.fieldpar[0][i][1]=np.random.uniform(np.amin(grids[2]),np.amax(grids[2]))
                        angleph=np.random.uniform(0,2*np.pi)
                        self.fieldpar[1][i][0]=module*np.cos(angleph)
                        self.fieldpar[1][i][1]=module*np.sin(angleph)
                    if parameters[0]==3:
                        self.fieldpar[0][i][0]=np.random.uniform(np.amin(grids[2]),np.amax(grids[2]))
                        self.fieldpar[0][i][1]=np.random.uniform(np.amin(grids[2]),np.amax(grids[2]))
                        self.fieldpar[0][i][2]=np.random.uniform(np.amin(grids[2]),np.amax(grids[2]))
                        angleth=np.random.uniform(0,np.pi)
                        angleph=np.random.uniform(0,2*np.pi)
                        self.fieldpar[1][i][0]=module*np.sin(angleth)*np.cos(angleph)
                        self.fieldpar[1][i][1]=module*np.sin(angleth)*np.sin(angleph)
                        self.fieldpar[1][i][2]=module*np.cos(angleth)
            if parameters[6][1][0]=='from-file':
                self.fieldpar=np.load(parameters[6][1][1])
        
        #Store quantities to be used for the particle density
        self.__dim=parameters[0]
        self.densassign=parameters[7][1]
        self.__dx=grids[0]
        if parameters[0]==1:
            self.__unigrid=grids[2]
        if parameters[0]==2:
            self.__unigrid=grids[4][0]
        if parameters[0]==3:
            self.__unigrid=grids[6][0][0]
        self.__particlenumber=parameters[7][0]
        self.__totalmass=parameters[3]
        self.__condfrac=parameters[5]
        
        
        #Set the condensate
        if parameters[6][0][0]=='None':
            if parameters[0]==1:
                self.field=np.zeros((parameters[2]), dtype=complex)
            if parameters[0]==2:
                self.field=np.zeros((parameters[2],parameters[2]), dtype=complex)
            if parameters[0]==3:
                self.field=np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex)
        if parameters[6][0][0]=='testing':
            if parameters[0]==1:
                self.field=np.zeros((parameters[2]), dtype=complex)+1.0+1.0j
            if parameters[0]==2:
                #self.field=np.zeros((parameters[2],parameters[2]), dtype=complex)+1.0+1.0j
                rad=0.6
                self.field=np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*3))/(pow(1+0.091*((pow(grids[2]-1.25,2)+pow(grids[4]+0.4,2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*3))/(pow(1+0.091*((pow(grids[2]+1.25,2)+pow(grids[4]-0.3,2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*3))/(pow(1+0.091*((pow(grids[2]+1.8,2)+pow(grids[4]+2,2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))#+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*4))/(pow(1+0.091*((pow(grids[2]-2,2)+pow(grids[4]-3,2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))      
            if parameters[0]==3:
                rad=0.6
                self.field=np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*3))/(pow(1+0.091*((pow(grids[2]-1.25,2)+pow(grids[4]+0.4,2)+pow(grids[6],2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2],parameters[2])))+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*3))/(pow(1+0.091*((pow(grids[2]+1.25,2)+pow(grids[4]-0.3,2)+pow(grids[6],2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2],parameters[2])))+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(rad,3)*3))/(pow(1+0.091*((pow(grids[2]+1.8,2)+pow(grids[4]+2,2)+pow(grids[6],2))/(pow(rad,2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2],parameters[2])))
        if parameters[6][0][0]=='random':
            if parameters[0]==1:
                self.field=np.random.random((parameters[2]))+np.random.random((parameters[2]))*1.0j
            if parameters[0]==2:
                self.field=np.random.random((parameters[2],parameters[2]))+np.random.random((parameters[2],parameters[2]))*1.0j
            if parameters[0]==3:
                self.field=np.random.random((parameters[2],parameters[2],parameters[2]))+np.random.random((parameters[2],parameters[2],parameters[2]))*1.0j
        if parameters[6][0][0]=='centralsoliton':
            if parameters[0]==1:
                self.field=np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(parameters[6][0][1],3)))/(pow(1+0.091*((pow(grids[2],2))/(pow(parameters[6][0][1],2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))
            if parameters[0]==2:
                self.field=np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(parameters[6][0][1],3)))/(pow(1+0.091*((pow(grids[2],2)+pow(grids[4],2))/(pow(parameters[6][0][1],2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))
            if parameters[0]==3:
                self.field=np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(parameters[6][0][1],3)))/(pow(1+0.091*((pow(grids[2],2)+pow(grids[4],2)+pow(grids[6],2))/(pow(parameters[6][0][1],2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2],parameters[2])))
        if parameters[6][0][0]=='multiple-solitons':
            if parameters[0]==1:
                self.field=np.zeros((parameters[2]), dtype=complex)
                for i in range(int(parameters[6][0][2])):
                    self.field=self.field+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(parameters[6][0][1],3)*parameters[6][0][2]))/(pow(1+0.091*((pow(grids[2]-np.random.uniform(-(parameters[1]/2.0-1.5), (parameters[1]/2.0-1.5)),2))/(pow(parameters[6][0][1],2))),8)))*np.exp(-1.0j*np.random.random((parameters[2])))
            if parameters[0]==2:
                self.field=np.zeros((parameters[2],parameters[2]), dtype=complex)
                for i in range(int(parameters[6][0][2])):
                    self.field=self.field+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(parameters[6][0][1],3)*parameters[6][0][2]))/(pow(1+0.091*((pow(grids[2]-np.random.uniform(-(parameters[1]/2.0-1.5),(parameters[1]/2.0-1.5)),2)+pow(grids[4]-np.random.uniform(-(parameters[1]/2.0-1.5),(parameters[1]/2.0-1.5)),2))/(pow(parameters[6][0][1],2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))
            if parameters[0]==3:
                self.field=np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex)
                for i in range(int(parameters[6][0][2])):
                    self.field=self.field+np.sqrt(((parameters[3]*1024*pow(0.091,1.5))/(33*pow(np.pi,2)*pow(parameters[6][0][1],3)*parameters[6][0][2]))/(pow(1+0.091*((pow(grids[2]-np.random.uniform(-(parameters[1]/2.0-1.5),(parameters[1]/2.0-1.5)),2)+pow(grids[4]-np.random.uniform(-(parameters[1]/2.0-1.5),(parameters[1]/2.0-1.5)),2)+pow(grids[6]-np.random.uniform(-(parameters[1]/2.0-1.5),(parameters[1]/2.0-1.5)),2))/(pow(parameters[6][0][1],2))),8)))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2],parameters[2])))
        if parameters[6][0][0]=='from-file':
            self.field=np.load(parameters[6][0][1]) 
            
            
        #defining the attributes real and imaginary part for the condensate
        self.re=self.field.real
        self.imag=self.field.imag
        
        #normalizing the condensate
        if parameters[5]>0:
            integnorm=np.sum(np.abs(self.field)**2)*(grids[0]**parameters[0])
            self.field=self.field*np.sqrt(1/integnorm)
    
    #Density of the condensate
    @property
    def density(self):    
        return self.getcondfrac()*self.gettotalmass()*np.abs(self.field)**2
    
    #Phase of the condensate
    @property
    def phase(self):
        return np.angle(self.field)
    
    #Functions to get the stored elements for the field
    def getdim(self):
        return self.__dim
    
    def getdx(self):
        return self.__dx
    
    def getunigrid(self):
        return self.__unigrid
    
    def gettotalmass(self):
        return self.__totalmass
    
    def getcondfrac(self):
        return self.__condfrac
    
    def cellgrid(self):
        centercellgrid=(self.getunigrid()[1:]+self.getunigrid()[:-1])/2.0
        centercellgrid=np.append(np.concatenate(([centercellgrid[0]-self.getdx()],centercellgrid)),centercellgrid[-1]+self.getdx())
        return centercellgrid
    
    #Function that gives the weights for the elements of the grid for the particle density assignation
    @property
    def weightassign(self):    
        if self.densassign=='nearestgridpoint':
            indicer=np.digitize(self.fieldpar[0],self.cellgrid())-1
            weighter=np.multiply.reduce(self.fieldpar[0]/self.fieldpar[0],axis=1)
            poss=self.getunigrid()[indicer]
        
        if self.densassign=='cloudincell':
            indicer=np.zeros((2**self.getdim(),len(self.fieldpar[0]),self.getdim()), dtype=int)
            indicer[0]=np.digitize(self.fieldpar[0],self.getunigrid())
            indicer[indicer==len(self.getunigrid())]=0
            if self.getdim()==1:
                indicer[1]=indicer[0]-[1]
            if self.getdim()==2:
                indicer[1]=indicer[0]-[1,0]
                indicer[2]=indicer[0]-[0,1]
                indicer[3]=indicer[0]-[1,1]
            if self.getdim()==3:
                indicer[1]=indicer[0]-[1,0,0]
                indicer[2]=indicer[0]-[0,1,0]
                indicer[3]=indicer[0]-[0,0,1]
                indicer[4]=indicer[0]-[1,1,0]
                indicer[5]=indicer[0]-[1,0,1]
                indicer[6]=indicer[0]-[0,1,1]
                indicer[7]=indicer[0]-[1,1,1]
            
            #poss=np.zeros((2**self.getdim(),len(self.fieldpar[0]),self.getdim()))
            weighter=np.zeros((2**self.getdim(),len(self.fieldpar[0])))
            poss=self.getunigrid()[indicer]
            dist=np.abs(poss-self.fieldpar[0])
            dist=np.minimum(dist,(np.amax(self.getunigrid())-np.amin(self.getunigrid())+self.getdx())-dist)
            weighter=np.multiply.reduce(1-dist/self.getdx(),axis=2)
            #for i in range(2**self.getdim()):
            #    poss[i]=self.getunigrid()[indicer[i]]
            #    dist=np.abs(poss[i]-self.fieldpar[0])
            #    dist=np.minimum(dist,(np.amax(self.getunigrid())-np.amin(self.getunigrid())+self.getdx())-dist)
            #    weighter[i]=np.multiply.reduce(1-dist/self.getdx(),axis=1)
        return indicer, weighter, poss

    #Density of the particles
    @property
    def densitypar(self):
        minpoint=np.amin(self.cellgrid())
        maxpoint=np.amax(self.cellgrid())
        binlen=len(self.getunigrid())
        indice, peso, poss = self.weightassign
        
        if self.densassign=='nearestgridpoint':
            if self.getdim()==1:
                dens, griddens=np.histogramdd(self.fieldpar[0], bins=(binlen),range=[[minpoint, maxpoint]], weights=peso)
            if self.getdim()==2:
                 dens, griddens=np.histogramdd(self.fieldpar[0], bins=(binlen,binlen),range=[[minpoint, maxpoint], [minpoint, maxpoint]], weights=peso)
            if self.getdim()==3:
                dens, griddens=np.histogramdd(self.fieldpar[0], bins=(binlen,binlen,binlen),range=[[minpoint, maxpoint], [minpoint, maxpoint],[minpoint, maxpoint]], weights=peso)
        
        if self.densassign=='cloudincell':
            if self.getdim()==1:
                dens=np.zeros((binlen))
                for i in range(2**self.getdim()):
                    denstmp, griddens=np.histogramdd(poss[i], bins=(binlen),range=[[minpoint, maxpoint]], weights=peso[i])
                    dens=dens+denstmp
            if self.getdim()==2:
                dens=np.zeros((binlen,binlen))
                for i in range(2**self.getdim()):
                    denstmp, griddens=np.histogramdd(poss[i], bins=(binlen,binlen),range=[[minpoint, maxpoint], [minpoint, maxpoint]], weights=peso[i])
                    dens=dens+denstmp
            if self.getdim()==3:
                dens=np.zeros((binlen,binlen,binlen))
                for i in range(2**self.getdim()):
                    denstmp, griddens=np.histogramdd(poss[i], bins=(binlen,binlen,binlen),range=[[minpoint, maxpoint], [minpoint, maxpoint], [minpoint, maxpoint]], weights=peso[i])
                    dens=dens+denstmp
        
        #normalization
        if self.getcondfrac()<1:
            dens=(((1-self.getcondfrac())*self.gettotalmass())/(len(self.fieldpar[0])*self.getdx()**self.getdim()))*dens
        
        return dens

    

    
    

    
#spin-2 field
#initial conditions are given by ['testing','random','centralsoliton','from-file']
class spin2:
    def __init__(self, parameters, grids):
        #adjust the field according to dimensions
        if parameters[0]==1:
            if parameters[3][1][0]=='testing':
                self.field=np.zeros((parameters[2]), dtype=complex)
            if parameters[3][1][0]=='random':
                self.field=np.zeros((parameters[2]), dtype=complex)
            if parameters[3][1][0]=='centralsoliton':
                self.field=np.zeros((parameters[2]), dtype=complex)
            if parameters[3][1][0]=='from-file':
                self.field=np.load(parameters[3][1][1])
        
        if parameters[0]==2:
            self.field=np.array([[np.zeros((parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2]), dtype=complex)],[np.zeros((parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2]), dtype=complex)]])
            if parameters[3][1][0]=='testing':
                for i in range(parameters[0]):
                    for j in range(parameters[0]):
                        if i!=j:
                            self.field[i][j]=self.field[i][j]+1.0
            if parameters[3][1][0]=='random':
                for i in range(parameters[0]):
                    for j in range(parameters[0]):
                        if i!=j:
                            self.field[i][j]=self.field[i][j]+np.random.random((parameters[2],parameters[2]))+np.random.random((parameters[2],parameters[2]))*1.0j
            if parameters[3][1][0]=='centralsoliton':
                for i in range(parameters[0]):
                    for j in range(parameters[0]):
                        if i!=j:
                            self.field[i][j]=self.field[i][j]+0.5*np.sqrt(parameters[3][1][1]*(1/(1+0.091*(grids[2]**2+grids[4]**2)/parameters[3][1][2]**2)**8))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2])))
            if parameters[3][1][0]=='from-file':
                self.field=np.load(parameters[3][1][1])
        
        if parameters[0]==3:
            self.field=np.array([[np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex)],[np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex)],[np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex),np.zeros((parameters[2],parameters[2],parameters[2]), dtype=complex)]])
            if parameters[3][1][0]=='testing':
                for i in range(parameters[0]):
                    for j in range(parameters[0]):
                        if i!=j:
                            self.field[i][j]=self.field[i][j]+1.0
            if parameters[3][1][0]=='random':
                for i in range(parameters[0]):
                    for j in range(parameters[0]):
                        if i!=j:
                            self.field[i][j]=self.field[i][j]+np.random.random((parameters[2],parameters[2],parameters[2]))*1.0j
            if parameters[3][1][0]=='centralsoliton':
                for i in range(parameters[0]):
                    for j in range(parameters[0]):
                        if i!=j:
                            self.field[i][j]=self.field[i][j]+(1.0/6.0)*np.sqrt(parameters[3][1][1]*(1/(1+0.091*(grids[2]**2+grids[4]**2+grids[6]**2)/parameters[3][1][2]**2)**8))*np.exp(-1.0j*np.random.random((parameters[2],parameters[2],parameters[2])))
            if parameters[3][1][0]=='from-file':
                self.field=np.load(parameters[3][1][1])
                             
        #defining the attributes real and imaginary part and dimension
        self.re=self.field.real
        self.imag=self.field.imag
        self.__dim=parameters[0]
        
        #normalizing
        if parameters[0]==2:
            integnorm=np.sum(np.abs(self.field[0][1])**2+np.abs(self.field[1][0])**2)*(grids[0]**parameters[0])
            self.field=self.field*np.sqrt(1/integnorm)
        if parameters[0]==3:
            integnorm=np.sum(np.abs(self.field[0][1])**2+np.abs(self.field[0][2])**2+np.abs(self.field[1][0])**2+np.abs(self.field[1][2])**2+np.abs(self.field[2][0])**2+np.abs(self.field[2][1])**2)*(grids[0]**parameters[0])
            self.field=self.field*np.sqrt(1/integnorm)
    
    def getdim(self):
        return self.__dim
    
    @property
    def density(self):
        if self.getdim()==1:
            return np.abs(self.field)**2
        if self.getdim()==2:
            return np.abs(self.field[0][1])**2+np.abs(self.field[1][0])**2
        if self.getdim()==3:
            return np.abs(self.field[0][1])**2+np.abs(self.field[0][2])**2+np.abs(self.field[1][0])**2+np.abs(self.field[1][2])**2+np.abs(self.field[2][0])**2+np.abs(self.field[2][1])**2
    
    @property
    def phase(self):
        return np.angle(self.field)