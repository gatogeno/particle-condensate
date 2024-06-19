#Load the modules needed
import numpy as np
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
import yt
from utilities import storer, grid, fields

import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
import matplotlib.animation as animation
from IPython.display import HTML

#Energy plots
def plotenergy(parameters, timeline, energy):
    if parameters[5]==1 or parameters[5]==0:
        figdcompfin, ((ax1dcompfin, ax2dcompfin), (ax3dcompfin, ax4dcompfin)) = plt.subplots(2, 2, figsize=(8,6))
        figdcompfin.suptitle('Energy plots')
        ax1dcompfin.set_xlabel("t")
        ax1dcompfin.set_ylabel("Energy")
        ax1dcompfin.plot(timeline,energy[2][0],label="total energy")
        ax1dcompfin.plot(timeline,energy[2][1],'--',label="kinetic")
        ax1dcompfin.plot(timeline,energy[2][2],'--',label="potential")
        ax1dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax1dcompfin.legend()
        ax2dcompfin.set_xlabel("t")
        ax2dcompfin.set_ylabel("Kinetic Energy")
        ax2dcompfin.plot(timeline,energy[2][1])
        ax2dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax3dcompfin.set_xlabel("t")
        ax3dcompfin.set_ylabel("Potential Energy")
        ax3dcompfin.plot(timeline,energy[2][2])
        ax3dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax4dcompfin.set_xlabel("t")
        ax4dcompfin.set_ylabel("Total Energy")
        ax4dcompfin.plot(timeline,energy[2][0])
        ax4dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        figdcompfin.tight_layout(pad=2.0)
        shower=plt.show()
    else:
        figdcompfin, ((ax1dcompfin, ax2dcompfin), (ax3dcompfin, ax4dcompfin)) = plt.subplots(2, 2, figsize=(8,6))
        figdcompfin.suptitle('Energy plots')
        ax1dcompfin.set_xlabel("t")
        ax1dcompfin.set_ylabel("Total energy")
        ax1dcompfin.plot(timeline,energy[2][0],label="total")
        ax1dcompfin.plot(timeline,energy[0][1],'--',label="kinetic particle")
        ax1dcompfin.plot(timeline,energy[0][2],'--',label="potential particle")
        ax1dcompfin.plot(timeline,energy[1][1],'--',label="kinetic condensate")
        ax1dcompfin.plot(timeline,energy[1][2],'--',label="potential condensate")
        ax1dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax1dcompfin.legend()
        ax2dcompfin.set_xlabel("t")
        ax2dcompfin.set_ylabel("Kinetic Energy")
        ax2dcompfin.plot(timeline,energy[2][1],label="total")
        ax2dcompfin.plot(timeline,energy[0][1],'--',label="particles")
        ax2dcompfin.plot(timeline,energy[1][1],'--',label="condensate")
        ax2dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax2dcompfin.legend()
        ax3dcompfin.set_xlabel("t")
        ax3dcompfin.set_ylabel("Potential Energy")
        ax3dcompfin.plot(timeline,energy[2][2],label="total")
        ax3dcompfin.plot(timeline,energy[0][2],'--',label="particles")
        ax3dcompfin.plot(timeline,energy[1][2],'--',label="condensate")
        ax3dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax3dcompfin.legend()
        ax4dcompfin.set_xlabel("t")
        ax4dcompfin.set_ylabel("Total Energy")
        ax4dcompfin.plot(timeline,energy[2][0],label="total")
        ax4dcompfin.plot(timeline,energy[0][0],'--',label="particle")
        ax4dcompfin.plot(timeline,energy[1][0],'--',label="condensate")
        ax4dcompfin.yaxis.get_major_formatter().set_useOffset(False)
        ax4dcompfin.legend()
        figdcompfin.tight_layout(pad=2.0)
        shower=plt.show()
    return shower

def plottotalmass(parameters,timeline,totalmass):
    if parameters[5]==1 or parameters[5]==0:
        fig, ax = plt.subplots(figsize=[4,4])
        fig.suptitle('Mass plot')
        ax.set_xlabel("t")
        ax.set_ylabel("Mass")
        ax.plot(timeline,totalmass[2])
        ax.yaxis.get_major_formatter().set_useOffset(False)
        shower=plt.show()
    else:
        fig, ax = plt.subplots(figsize=[4,4])
        fig.suptitle('Mass plot')
        ax.set_xlabel("t")
        ax.set_ylabel("Mass")
        ax.plot(timeline,totalmass[2],label="total mass")
        ax.plot(timeline,totalmass[0],'--',label="particle mass")
        ax.plot(timeline,totalmass[1],'--',label="condensate mass")
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.legend()
        shower=plt.show()
    return shower

def visor(dens,parameters,grids,h=-1):
    ncond=dens
    if parameters[0]==1:
        fig2 = plt.figure(figsize=(4,3.8))
        f2=plt.plot(grids[2], dens)
        shower=plt.show()
    if parameters[0]==2:
        fig2 = plt.figure(figsize=(4,3.8))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        f2=plt.scatter(grids[2], grids[4], c=dens, cmap='inferno')
        fig2.colorbar(f2)
        shower=plt.show()
    if parameters[0]==3:
        xdots=parameters[2]
        datab = dict(density = (ncond))
        bbox = np.array([[0, xdots], [0, xdots], [0, xdots]])
        ds = yt.load_uniform_grid(datab, ncond.shape, bbox=bbox, nprocs=parameters[2])
    
        sc = yt.create_scene(ds, ("gas", "density"))

        cam = sc.add_camera(ds)
        cam.position = np.array([0.0, -0.4, -0.1])
        sc.camera.resolution = 400
        sc.camera.width = (1.6)

        source = sc[0]
        source.set_field(("gas", "density"))
        source.set_log(False)
        source.set_use_ghost_zones(True)
    
        bounds = (np.amin(ncond), np.amax(ncond))
        tf = yt.ColorTransferFunction(bounds)
        tf.add_layers(10, colormap="RdBu")
        if np.amin(ncond)!=np.amax(ncond):
            def linramp(vals, minval, maxval):
                return (vals - vals.min()) / (vals.max() - vals.min())
            tf.map_to_colormap(np.amin(ncond), np.amax(ncond), colormap="inferno", scale_func=linramp)
        source.tfh.tf = tf
        source.tfh.bounds = bounds
       
        sc.annotate_axes(alpha=0.1)
        sc.annotate_domain(ds, color=[1, 1, 1, 0.1]) #Dibuja el borde del cubo, el cuarto valor es su opacidad
    
        if h>=0:
            shower=None
            sc.save_annotated(f'{parameters[6]}/visualizacion_'+ str(h)+'.jpg',sigma_clip=6.0)
        else:
            shower=sc.show(sigma_clip=6.0)    
    return shower

#Visualization by cuts for 3D objects
def visorcuts(dens,grids):
    rho=dens
    x=grids[2]
    y=grids[4]
    z=grids[6]
    plz=int(np.ceil((len(x)-1)/2))
    plz2=int(np.ceil((len(x)-1)/5))
       
    fig2 = plt.figure(figsize=(20,5))
    ax2a = fig2.add_subplot(131)
    ax2a.set_title("Plane (x, y, %.2f)" % z[plz2,plz2,plz2])
    ax2a.set_xlabel('$x$')
    ax2a.set_ylabel('$y$')
    f2a=ax2a.scatter(x[:,:,plz2], y[:,:,plz2], c=rho[:,:,plz2], cmap='inferno')
        
    ax2b = fig2.add_subplot(132)
    ax2b.set_title("Plane (x, y, %.2f)" % z[plz,plz,plz])
    ax2b.set_xlabel('$x$')
    ax2b.set_ylabel('$y$')
    f2b=ax2b.scatter(x[:,:,plz], y[:,:,plz], c=rho[:,:,plz], cmap='inferno')
        
    ax2c = fig2.add_subplot(133)
    ax2c.set_title("Plane (x, y, %.2f)" % z[4*plz2,4*plz2,4*plz2])
    ax2c.set_xlabel('$x$')
    ax2c.set_ylabel('$y$')
    f2c=ax2c.scatter(x[:,:,4*plz2], y[:,:,4*plz2], c=rho[:,:,4*plz2], cmap='inferno')
        
    fig2.colorbar(f2a,ax=ax2a)
    fig2.colorbar(f2b,ax=ax2b)
    fig2.colorbar(f2c,ax=ax2c)
        
    shower=plt.show()
    return shower

#Densities by axes, centered in the middle point
def visordensity(dens,parameters,grids,ini=1):
    ncond=dens
    midpnt=int(parameters[2]/2)
    figdcompfin, (ax1dcompfin, ax2dcompfin, ax3dcompfin) = plt.subplots(1, 3, figsize=(12,4))
    if ini==1:
        figdcompfin.suptitle(f'Number densities by axes after 0 iterations')
    else:
        figdcompfin.suptitle(f'Number densities by axes after {parameters[9][1]} iterations')
    ax1dcompfin.set_xlabel("x")
    ax1dcompfin.set_ylabel("Density number")
    ax1dcompfin.plot(grids[2][:,midpnt,midpnt],ncond[:,midpnt,midpnt])
    ax2dcompfin.set_xlabel("y")
    ax2dcompfin.set_ylabel("Density number")
    ax2dcompfin.plot(grids[4][midpnt,:,midpnt],ncond[midpnt,:,midpnt])
    ax3dcompfin.set_xlabel("z")
    ax3dcompfin.set_ylabel("Density number")
    ax3dcompfin.plot(grids[6][midpnt,midpnt,:],ncond[midpnt,midpnt,:])
    figdcompfin.tight_layout(pad=2.0)
    shower=plt.show()
    return shower

#Plot of projected densities
def projection(dens,parameters,grids):
    ncond=dens
    leng=parameters[1]
    datab = dict(density = (ncond, "code_length"))
    bbox = np.array([[0, leng], [0, leng], [0, leng]])
    ds = yt.load_uniform_grid(datab, ncond.shape, length_unit="code_length", bbox=bbox, nprocs=parameters[2])
    
    slc=yt.ProjectionPlot(ds, "z", ("gas", "density"))
    slc.set_cmap(field=("gas", "density"), cmap="inferno")
    slc.set_figure_size(3)
    slc.hide_colorbar()
    slc.hide_axes()
    
    return slc

#Potential plots by axis, centered in the middlepoint.
def visorpot(pot,parameters,grids,ini=1):
    if parameters[0]==2:
        xx=grids[2]
        yy=grids[4]
        midpnt=int(parameters[2]/2)
        figpot, (ax1pot, ax2pot) = plt.subplots(1, 2, figsize=(7,3.5))
        ax1pot.set_xlabel("x")
        ax2pot.set_xlabel("y")
        if ini==1:
            figpot.suptitle(f'Potentials for the condensate by axes after 0 iterations')
        else:
            figpot.suptitle(f'Potentials for the condensate by axes after {parameters[9][1]} iterations')
        ax1pot.set_ylabel("Potential")
        ax2pot.set_ylabel("Potential")
        ax1pot.plot(xx[:,midpnt],pot[:,midpnt])   
        ax2pot.plot(yy[midpnt,:],pot[midpnt,:])
        figpot.tight_layout(pad=2.0)
    if parameters[0]==3:
        xx=grids[2]
        yy=grids[4]
        zz=grids[6]
        midpnt=int(parameters[2]/2)
        figpot, (ax1pot, ax2pot, ax3pot) = plt.subplots(1, 3, figsize=(12,4))
        ax1pot.set_xlabel("x")
        ax2pot.set_xlabel("y")
        ax3pot.set_xlabel("z")
        if ini==1:
            figpot.suptitle(f'Potentials for the condensate by axes after 0 iterations')
        else:
            figpot.suptitle(f'Potentials for the condensate by axes after {parameters[9][1]} iterations')
        ax1pot.set_ylabel("Potential")
        ax2pot.set_ylabel("Potential")
        ax3pot.set_ylabel("Potential")
        ax1pot.plot(xx[:,midpnt,midpnt],pot[:,midpnt,midpnt])   
        ax2pot.plot(yy[midpnt,:,midpnt],pot[midpnt,:,midpnt])     
        ax3pot.plot(zz[midpnt,midpnt,:],pot[midpnt,midpnt,:])
             
        figpot.tight_layout(pad=2.0)
    shower=plt.show()
    return shower

#Histogram of particle velocities
def histogramvelocities(parameters,psi,psiini,binnum):
    if parameters[5]<1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
        ax1.set_xlabel("|v|")
        ax2.set_xlabel("|v|")
        ax1.set_title('Initial distribution of velocities')
        ax2.set_title('Final distribution of velocities')
        ax1.set_ylabel("Number of particles")
        ax2.set_ylabel("Number of particles")
        ax1.hist(np.sqrt(np.sum(psiini.fieldpar[1]**2,axis=1)), bins=binnum)   
        ax2.hist(np.sqrt(np.sum(psi.fieldpar[1]**2,axis=1)), bins=binnum)
        fig.tight_layout(pad=2.0)
        shower=plt.show()
        return shower

#Animation of densities and phase:
def animaplots(parameters,grids,psi,selector,save=False):
    solutions=[]
    for i in range(int(parameters[9][1]/parameters[9][2])+1):
        solutions.append([])
    for i in range(int(parameters[9][1]/parameters[9][2])):
        if selector==0:
            parameters[6][1]=['from-file',parameters[10]+'/evolution/fieldpar_'+str(parameters[9][2]*i)+'.npy']
            psitemp=fields.spin0(parameters,grids)
            solutions[i]=psitemp.densitypar
        if selector==1:
            parameters[6][0]=['from-file',parameters[10]+'/evolution/field_'+str(parameters[9][2]*i)+'.npy']
            psitemp=fields.spin0(parameters,grids) 
            solutions[i]=psitemp.density
        if selector==2:
            parameters[6][0]=['from-file',parameters[10]+'/evolution/field_'+str(parameters[9][2]*i)+'.npy']
            psitemp=fields.spin0(parameters,grids)
            solutions[i]=psitemp.phase
    if selector==0:
        solutions[int(parameters[9][1]/parameters[9][2])]=psi.densitypar
    if selector==1:
        solutions[int(parameters[9][1]/parameters[9][2])]=psi.density
    if selector==2:
        solutions[int(parameters[9][1]/parameters[9][2])]=psi.phase
        
    plt.ioff()
    if parameters[0]==1:
        fig = plt.figure(figsize=(4,4))
    if parameters[0]==2:
        fig = plt.figure(figsize=(4,4))
    if parameters[0]==3:
        fig = plt.figure(figsize=(8,8))
        gs = fig.add_gridspec(2,2, wspace=0.4)
            

    def animate(i):
        fig.clear()
        if parameters[0]==1:
            plt.xlabel('$x$')
            plt.ylabel(f'$|\phi|^2$')
            plt.ylim(0,np.amax(solutions))
            img=plt.plot(grids[2],solutions[i])
        if parameters[0]==2:
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            if selector==0:
                img=plt.scatter(grids[2], grids[4], c=solutions[i], cmap='inferno', edgecolors='none',vmin=0,vmax=np.amax(psi.densitypar))
            if selector==1:
                img=plt.scatter(grids[2], grids[4], c=solutions[i], cmap='inferno', edgecolors='none',vmin=0,vmax=np.amax(psi.density))
            if selector==2:
                img=plt.scatter(grids[2], grids[4], c=solutions[i], cmap='hsv', edgecolors='none',vmin=np.amin(psi.phase),vmax=np.amax(psi.phase))
            fig.colorbar(img)
        if parameters[0]==3:
            ax1 = fig.add_subplot(gs[0, :], projection='3d')
            ax1.set_title(f'Iteration = {parameters[9][2]*i}')
            if selector==0:
                img=ax1.scatter3D(grids[2], grids[4], grids[6], c=solutions[i], cmap='inferno', edgecolors='none',vmin=0,vmax=np.amax(psi.densitypar))
            if selector==1:
                img=ax1.scatter3D(grids[2], grids[4], grids[6], c=solutions[i], cmap='inferno', edgecolors='none',vmin=0,vmax=np.amax(psi.density))
            if selector==2:
                img=ax1.scatter3D(grids[2], grids[4], grids[6], c=solutions[i], cmap='hsv', edgecolors='none',vmin=np.amin(psi.phase),vmax=np.amax(psi.phase))
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax2 = fig.add_subplot(gs[1, 0], projection='3d')
            if selector==0:
                img2=ax2.scatter3D(grids[2], grids[4], grids[6][0][0][int(np.ceil((len(grids[2])-1)/2))], c=solutions[i], cmap='inferno', edgecolors='none',vmin=0,vmax=np.amax(psi.densitypar))
            if selector==1:
                img2=ax2.scatter3D(grids[2], grids[4], grids[6][0][0][int(np.ceil((len(grids[2])-1)/2))], c=solutions[i], cmap='inferno', edgecolors='none',vmin=0,vmax=np.amax(psi.density))
            if selector==2:
                img2=ax2.scatter3D(grids[2], grids[4], grids[6][0][0][int(np.ceil((len(grids[2])-1)/2))], c=solutions[i], cmap='hsv', edgecolors='none',vmin=np.amin(psi.phase),vmax=np.amax(psi.phase))
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax3 = fig.add_subplot(gs[1, 1])
            if selector==0:
                img3= ax3.scatter(grids[2][:,:,int(np.ceil((len(grids[2])-1)/2))], grids[4][:,:,int(np.ceil((len(grids[2])-1)/2))], c = (solutions[i])[:,:,int(np.ceil((len(grids[2])-1)/2))], cmap='inferno',vmin=0,vmax=np.amax(psi.densitypar))
            if selector==1:
                img3= ax3.scatter(grids[2][:,:,int(np.ceil((len(grids[2])-1)/2))], grids[4][:,:,int(np.ceil((len(grids[2])-1)/2))], c = (solutions[i])[:,:,int(np.ceil((len(grids[2])-1)/2))], cmap='inferno',vmin=0,vmax=np.amax(psi.density))
            if selector==2:
                img3= ax3.scatter(grids[2][:,:,int(np.ceil((len(grids[2])-1)/2))], grids[4][:,:,int(np.ceil((len(grids[2])-1)/2))], c = (solutions[i])[:,:,int(np.ceil((len(grids[2])-1)/2))], cmap='hsv',vmin=np.amin(psi.phase),vmax=np.amax(psi.phase))
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')

            plt.colorbar(img,ax=ax1, orientation='horizontal',fraction=0.06)#, pad=0.1)

    ani = animation.FuncAnimation(fig, animate, interval=90, frames=range(len(solutions)), repeat=False)
    if save==True:
        if selector==0:
            ani.save(f'{parameters[10]}/densitypar.gif', writer='pillow')
        if selector==1:
            ani.save(f'{parameters[10]}/density.gif', writer='pillow')
        if selector==2:
            ani.save(f'{parameters[10]}/phase.gif', writer='pillow')
    HTML(ani.to_jshtml())
    return ani

#Matter power spectrum for overdensity
def plotpowerspectrum(parameters, grids, fieldY, binnum):
    #Compute the power spectrum
    averagedens=parameters[3]/parameters[1]**parameters[0]
    if parameters[5]==1:
        delta=(fieldY.density-averagedens)/(averagedens)
    elif parameters[5]==0:
        delta=(fieldY.densitypar-averagedens)/(averagedens)
    else:
        delta=(fieldY.density+fieldY.densitypar-averagedens)/(averagedens)
    deltak=fftshift(fftn(delta))
    Pk=np.abs(deltak)**2
    
    #Create the bins
    k_bins = np.linspace(0, np.amax(grids[1]), binnum)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    power_spectrum_bins = np.zeros(len(k_bin_centers))
    bin_counts = np.zeros(len(k_bin_centers))
    
    #Average over spherical shells
    if parameters[0]==1:
        for i in range(parameters[2]):
            k_value = grids[1][i]
            p_value = Pk[i]
            bin_index = np.digitize(k_value, k_bins) - 1
            if bin_index >= 0 and bin_index < len(power_spectrum_bins):
                power_spectrum_bins[bin_index] += p_value
                bin_counts[bin_index] += 1
    if parameters[0]==2:
        for i in range(parameters[2]):
            for j in range(parameters[2]):
                k_value = grids[1][i, j]
                p_value = Pk[i, j]
                bin_index = np.digitize(k_value, k_bins) - 1
                if bin_index >= 0 and bin_index < len(power_spectrum_bins):
                    power_spectrum_bins[bin_index] += p_value
                    bin_counts[bin_index] += 1
    if parameters[0]==3:
        for i in range(parameters[2]):
            for j in range(parameters[2]):
                for l in range(parameters[2]):
                    k_value = grids[1][i, j, l]
                    p_value = Pk[i, j, l]
                    bin_index = np.digitize(k_value, k_bins) - 1
                    if bin_index >= 0 and bin_index < len(power_spectrum_bins):
                        power_spectrum_bins[bin_index] += p_value
                        bin_counts[bin_index] += 1

    # Avoid division by zero
    bin_counts[bin_counts == 0] = 1
    power_spectrum_bins /= bin_counts
    
    #Generate the plot
    plt.loglog(k_bin_centers, power_spectrum_bins)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    plt.title('Matter Power Spectrum')
    shower=plt.show()
    
    return shower

#Density plot radial
def plotdensityrad(parameters, grids, fieldY, binnum,logplot=False):
    #Identifies the highest value of the density and computes the distance of the coordinates to its coordinates
    if parameters[5]==1:
        center_index=np.unravel_index(np.argmax(fieldY.density), fieldY.density.shape)
    elif parameters[5]==0:
        center_index=np.unravel_index(np.argmax(fieldY.densitypar), fieldY.densitypar.shape)
    else:
        center_index=np.unravel_index(np.argmax(fieldY.density+fieldY.densitypar), (fieldY.density+fieldY.densitypar).shape)
    
    if parameters[0]==1:
        radialdist = np.sqrt((grids[2] - grids[2][center_index[0]])**2)
    if parameters[0]==2:
        radialdist = np.sqrt((grids[2] - grids[2][center_index[0],center_index[1]])**2 + (grids[4] - grids[4][center_index[0],center_index[1]])**2)
    if parameters[0]==3:
        radialdist = np.sqrt((grids[2] - grids[2][center_index[0],center_index[1],center_index[2]])**2 + (grids[4] - grids[4][center_index[0],center_index[1],center_index[2]])**2 + (grids[6] - grids[6][center_index[0],center_index[1],center_index[2]])**2)
    
    #Create bins
    bins = np.linspace(0, np.amax(radialdist),int(binnum))
    r_bin_centers = 0.5 * (bins[1:] + bins[:-1])
    density_bins = np.zeros(len(r_bin_centers))
    densitypar_bins = np.zeros(len(r_bin_centers))
    counts = np.zeros(len(r_bin_centers))
    countspar = np.zeros(len(r_bin_centers))
    
    if parameters[0]==1:
        for i in range(parameters[2]):
            r_value = radialdist[i]
            rho_value = fieldY.density[i]
            rhopar_value = fieldY.densitypar[i]
            bin_index = np.digitize(r_value, r_bin_centers)-1
            if 0 <= bin_index < len(density_bins):
                density_bins[bin_index] += rho_value
                counts[bin_index] += 1
            if 0 <= bin_index < len(densitypar_bins):
                densitypar_bins[bin_index] += rhopar_value
                countspar[bin_index] += 1
    if parameters[0]==2:
        for i in range(parameters[2]):
            for j in range(parameters[2]):
                r_value = radialdist[i,j]
                rho_value = fieldY.density[i,j]
                rhopar_value = fieldY.densitypar[i,j]     
                bin_index = np.digitize(r_value, r_bin_centers)-1
                if 0 <= bin_index < len(density_bins):
                    density_bins[bin_index] += rho_value
                    counts[bin_index] += 1
                if 0 <= bin_index < len(densitypar_bins):
                    densitypar_bins[bin_index] += rhopar_value
                    countspar[bin_index] += 1
    if parameters[0]==3:
        for i in range(parameters[2]):
            for j in range(parameters[2]):
                for l in range(parameters[2]):
                    r_value = radialdist[i,j,l]                 
                    rho_value = fieldY.density[i,j,l]
                    rhopar_value = fieldY.densitypar[i,j,l]      
                    bin_index = np.digitize(r_value, r_bin_centers)-1
                    if 0 <= bin_index < len(density_bins):
                        density_bins[bin_index] += rho_value
                        counts[bin_index] += 1
                    if 0 <= bin_index < len(densitypar_bins):
                        densitypar_bins[bin_index] += rhopar_value
                        countspar[bin_index] += 1
        
    # Avoid division by zero
    counts[counts == 0] = 1
    density_bins /= counts
    countspar[countspar == 0] = 1
    densitypar_bins /= countspar
    
    #Create the plot
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\rho$')
    plt.title('Density')
    if logplot==True:
        plt.loglog(r_bin_centers, density_bins+densitypar_bins,label="Total density")
        if parameters[5]>0 and parameters[5]<1:
            plt.loglog(r_bin_centers, density_bins,'--',label="Density condensate")
            plt.loglog(r_bin_centers, densitypar_bins,'--',label="Density particles")
            plt.legend()
    else:
        plt.plot(r_bin_centers, density_bins+densitypar_bins,label="Total density")
        if parameters[5]>0 and parameters[5]<1:
            plt.plot(r_bin_centers, density_bins,'--',label="Density condensate")
            plt.plot(r_bin_centers, densitypar_bins,'--',label="Density particles")
            plt.legend()
    shower=plt.show()
    
    return shower