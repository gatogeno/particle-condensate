import numpy as np
from scipy.fftpack import fftn, fftfreq, ifftn, fftshift, ifftshift

#Create a spatial and momentum grid
def creategrid(parameters):
    dx = parameters[1]/parameters[2]
    xlin = np.linspace(-parameters[1]/2.0, parameters[1]/2.0, parameters[2],endpoint=False)
    klin = fftshift(2*np.pi*fftfreq(parameters[2], dx))
    
    if parameters[0]==1:
        ksq=klin**2
        return dx, ksq, xlin, klin
    elif parameters[0]==2:
        xx, yy = np.meshgrid(xlin,xlin, indexing='ij')
        kx, ky = np.meshgrid(klin,klin, indexing='ij')
        ksq=kx**2+ky**2
        return dx, ksq, xx, kx, yy, ky
    elif parameters[0]==3:
        xx, yy, zz = np.meshgrid(xlin,xlin,xlin, indexing='ij')
        kx, ky, kz = np.meshgrid(klin,klin,klin, indexing='ij')
        ksq=kx**2+ky**2+kz**2
        return dx, ksq, xx, kx, yy, ky, zz, kz