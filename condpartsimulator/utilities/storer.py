import sys
import os
import shutil
import numpy as np
from utilities import additionalfunctions as adf

#Function that reads the initial file and pass the parameters to the program
def reader(filename,secondread=False,display=True):
    #List of parameters needed to work
    parameterlist=['dimension','boxlength','sitesbylength','totalmass','fieldtype','potential','evolver','output']
    #List of field types supported
    fieldlist=['spin0','spin2']
    #List of initial conditions. Entry 0 contains the possible initial conditions, next entries are additional values needed for such initial condition.
    iclist=[['None'],['testing'],['random'],['centralsoliton','radc'],['multiple-solitons','radc','numbersolitons'],['from-file','initialfilef']]
    iclistth=[['None'],['testing'],['random'],['maxwellian','sigma'],['from-file','initialfileth']]
    #List of particle density assignments
    dalist=['nearestgridpoint','cloudincell']
    #List of potentials supported. The potential "testing" is a potential well and "full" contains the harmonical, gravitational and self-interaction, which can be turned on and off setting the parameters to zero in the harmonical and self-interaction and setting True or False for gravitational
    potentiallist=[['testing','lengthwell'],['full','gravitational','omega','selfcoupling']]
    #List of evolvers
    evollist=['kickdriftkick','driftkickdrift']
    
    #Open the initial file. If it doesn't exist it generates an error
    try:
        f=open(f'{filename}')
    except FileNotFoundError:
        print('Error opening input file')
    
    #Introduces the lines of the input file in a temporal vector called storertmp deleting empty lines and comments
    storertmp=[line for line in f.readlines() if not line.isspace()]
    storertmp=adf.remove_comments(storertmp)
    f.close()
    
    #create the array which will contain the parameters
    parameters=[]
    
    #Check if the parameters in the parameter list exist in the initial file
    for i in range(0,len(parameterlist)):
        element=adf.checkerlist(parameterlist[i],storertmp)
        
        #Check particular conditions in the parameters and it adds the values in the array parameters       
        #Dimension. Only works if D=1,2 or 3
        if i==0:
            if int(element)>3 or int(element)<1:
                print(f'Code not supported for d={int(element)}. Try with dimensions 1, 2 or 3.')
                break
            else:
                parameters.append(int(element))
        
        #Box length L, it is a cubic box
        if i==1:
            if float(element)<=0:
                print(f'The length must be a positive number.')
                break
            else:
                parameters.append(float(element))
        
        #Number of sites by length (N)
        if i==2:
            if int(element)<=0:
                print(f'The number of sites per length must be a positive number.')
                break
            else:
                parameters.append(int(element))
            
        #Total mass M in the system
        if i==3:
            if float(element)<=0:
                print(f'The total mass must be a positive number.')
                break
            else:
                parameters.append(float(element))
                
        #Fields configuration to be simulated
        if i==4:
            #It adds the name of the type of field to the array
            counteriffield=0
            for j in range(len(fieldlist)):
                if element==fieldlist[j]:
                    counteriffield=counteriffield+1
                    indexfield=j
            if counteriffield==0:
                print('Field configuration not available. Here is the list of available fields:')
                for j in range(0,len(fieldlist)):
                    print(fieldlist[j])
                sys.exit()
            else:
                parameters.append(element)
                
            #Check the condensate fraction. If it is different than spin0 it assigns automatically f=1.
            if element=='spin0':
                elementcf=adf.checkerlist('condensatefraction',storertmp)
                if float(elementcf)<0 or float(elementcf)>1:
                    print(f'The condensate fraction is a number between 0 and 1, including both.')
                    break
                else:
                    condfraction=float(elementcf)
            else:
                condfraction=1.0
            parameters.append(condfraction)
            
            #It adds the initial condition for the chosen field to the array
            parameters.append([])
            parameters[6].append([])
            parameters[6].append([])
            counterific=0
            counterificth=0
            indexic=0
            indexicth=0
            if element=='spin0':
                if condfraction==1:
                    elementic=adf.checkerlist('initialcondf',storertmp)
                    elementicth='None'
                    counterificth=1
                elif condfraction==0:
                    elementic='None'
                    elementicth=adf.checkerlist('initialcondth',storertmp)
                    counterific=1
                else:
                    elementic=adf.checkerlist('initialcondf',storertmp)
                    elementicth=adf.checkerlist('initialcondth',storertmp)
            else:
                elementic=adf.checkerlist('initialcondf',storertmp)
                elementicth='None'
                counterificth=1
            
            for j in range(len(iclist)):
                if elementic==iclist[j][0]:
                    counterific=counterific+1
                    indexic=j
            for j in range(len(iclistth)):
                if elementicth==iclistth[j][0]:
                    counterificth=counterificth+1
                    indexicth=j
            
            if counterific==0:
                if element=='spin0':
                    print('Initial condition for condensate not available. Here is the list of available initial conditions:')
                else:
                    print('Initial condition for the field not available. Here is the list of available initial conditions:')
                for j in range(len(1,iclist)):
                    print(iclist[j][0])
                sys.exit()
            else:
                if counterificth==0:
                    print('Initial condition for particles not available. Here is the list of available initial conditions:')
                    for j in range(1,len(iclistth)):
                        print(iclistth[j][0])
                    sys.exit()
                else:
                    parameters[6][0].append(elementic)
                    parameters[6][1].append(elementicth)
                    
            #Here we add the parameters of the chosen initial condition
            for l in range(1,len(iclist[indexic])):
                elementparic=adf.checkerlist(iclist[indexic][l],storertmp)
                if elementparic=='from-file':
                    parameters[6][0].append(elementparic)
                else:
                    parameters[6][0].append(float(elementparic)) 
            for l in range(1,len(iclistth[indexicth])):
                elementparicth=adf.checkerlist(iclistth[indexicth][l],storertmp)
                if elementparicth=='from-file':
                    parameters[6][1].append(elementparicth)
                else:
                    parameters[6][1].append(float(elementparicth))
                    
            #It adds the number of test particles and the density assignment for the particles
            parameters.append([])
            if element=='spin0':
                if condfraction<1:
                    counterda=0
                    elementpart=adf.checkerlist('particlenumber',storertmp)
                    if int(elementpart)<=0:
                        print(f'The number of test particles must be a positive number.')
                        break
                    else:
                        parameters[7].append(int(elementpart))
                    elementpart=adf.checkerlist('densityassignment',storertmp)
                    for r in range(len(dalist)):
                        if elementpart==dalist[r]:
                            counterda=counterda+1
                    if counterda==0:
                        print(f'Density assignment for particles not available. Here is the list of available density asignments:')
                        for l in range(len(dalist)):
                            print(dalist[l])
                        sys.exit()
                    else:
                        parameters[7].append(elementpart)
                else:
                    parameters[7].append(0)
                    parameters[7].append('nearestgridpoint')
            else:
                parameters[7].append(0)
                parameters[7].append('nearestgridpoint')
                     
        #Potential
        if i==5:
            parameters.append([])
            indexpot=0
            counterpot=0
            for j in range(len(potentiallist)):
                if element==potentiallist[j][0]:
                    counterpot=counterpot+1
                    indexpot=j
            if counterpot==0:
                print('Potential not available. Here is the list of available potentials:')
                for j in range(len(potentiallist)):
                    print(potentiallist[j][0])
                sys.exit()
            else:
                parameters[8].append(element)
            #Here we add the parameters of the chosen potential
            if element==potentiallist[indexpot][0]:  
                for k in range(1,len(potentiallist[indexpot])):
                    elementpot=adf.checkerlist(potentiallist[indexpot][k],storertmp)
                    if potentiallist[indexpot][k]=='gravitational':
                        if elementpot.lower()=='true':
                            parameters[8].append(True)
                        elif elementpot.lower()=='false':
                            parameters[8].append(False)
                        else:
                            print('For the parameter "gravitational" the options are True if the gravitational potential is present or False if not')
                    else:
                        parameters[8].append(float(elementpot))
                        
        #Evolver
        if i==6:
            parameters.append([])
            counterevol=0
            for j in range(0,len(evollist)):
                if element==evollist[j]:
                    counterevol=counterevol+1
            if counterevol==0:
                print('Evolver not available. Here is the list of available evolvers:')
                for j in range(0,len(evollist)):
                    print(evollist[j][0])
                sys.exit()
            else:
                parameters[9].append(element)
                element2=adf.checkerlist('iterations',storertmp)
                parameters[9].append(int(element2))
                element2=adf.checkerlist('recordevery',storertmp)
                parameters[9].append(int(element2))
                if element==evollist[0] or element==evollist[1]:
                    element3=adf.checkerlist('imagprop',storertmp)
                    if element3.lower()=='true':
                        parameters[9].append(True)
                    elif element3.lower()=='false':
                        parameters[9].append(False)
                    else:
                        print('Please indicate True if the evolution is in imaginary time or False if the evolution is in real time')
                    
        #Output file name, create it if it doesn't exist
        if i==7:
            if secondread==False:
                isExist=os.path.exists(element)
                isExist2=os.path.exists(element+'/evolution')
                if not isExist:
                    os.makedirs(element)
                if not isExist2:
                    os.makedirs(element+'/evolution')
                parameters.append(element)
                shutil.copy2(filename, element+'/'+filename) 
            if secondread==True:
                parameters.append(element)
            
    
    #Print the parameters used in the simulation
    if display==True:
        print("*-----------------------------------*")
        print(f'Dimension: {parameters[0]}')
        print(f'Box length: {parameters[1]}')
        print(f'N: {parameters[2]}')
        print(f'Total mass: {parameters[3]}')
        print(f' ')
        print(f'Field configuration: {parameters[4]}')
        if parameters[4]=='spin0':
            print(f'Condensate fraction: {parameters[5]}')
            if parameters[5]==1:
                print(f'Initial configuration condensate: {parameters[6][0][0]}')
                for i in range(1,len(iclist[indexic])):
                    print(f'  {iclist[indexic][i]}: {parameters[6][0][i]}')
            elif parameters[5]==0:
                print(f'Initial configuration particles: {parameters[6][1][0]}')
                for i in range(1,len(iclistth[indexicth])):
                    print(f'  {iclistth[indexicth][i]}: {parameters[6][1][i]}')
                print(f'Number of test particles: {parameters[7][0]}')
                print(f'Density assignment: {parameters[7][1]}')
            else:
                print(f'Initial configuration condensate: {parameters[6][0][0]}')
                for i in range(1,len(iclist[indexic])):
                    print(f'  {iclist[indexic][i]}: {parameters[6][0][i]}')
                print(f'Initial configuration particles: {parameters[6][1][0]}')
                for i in range(1,len(iclistth[indexicth])):
                    print(f'  {iclistth[indexicth][i]}: {parameters[6][1][i]}')
                print(f'Number of test particles: {parameters[7][0]}')
                print(f'Density assignment: {parameters[7][1]}')
        
        else:
            print(f'Initial configuration: {parameters[6][0][0]}')
            for i in range(1,len(iclist[indexic])):
                print(f'  {iclist[indexic][i]}: {parameters[6][0][i]}')
        print(f' ')
        print(f'Potential type: {parameters[8][0]}')
        print(f'Potential parameters:')
        for i in range(1,len(potentiallist[indexpot])):
            print(f'  {potentiallist[indexpot][i]}: {parameters[8][i]}')
        print(f' ')
        print(f'Evolver: {parameters[9][0]}')
        print(f'Number of iterations: {parameters[9][1]}')
        print(f'Saving every: {parameters[9][2]} steps')
        if parameters[9][0]==evollist[0] or parameters[9][0]==evollist[1]:
            print(f'Imaginary time propagation: {parameters[9][3]}')
        print(f' ')
        print(f'Output files in: {parameters[10]}/')
        print("*-----------------------------------*")
    
    return parameters
    