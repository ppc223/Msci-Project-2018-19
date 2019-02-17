import matplotlib.pyplot as plt
import numpy as np
import scipy

from mpl_toolkits.mplot3d import Axes3D
from scipy.special import airy
from matplotlib import cm
from qutip import *

def myplotwigner(psi, xrange = [0,3], yrange = [-20,20], step = 0.1, title = 'Wigner Function'):
    """
    Function for plotting the Wiger function of a state which gives more control
    over the appearance of the graph than the built in qutip plot_wigner function
    """
    # Generate X and Y values from inputs
    xvec = np.arange(xrange[0], xrange[1], step)
    yvec = np.arange(yrange[0], yrange[1], step)
    X,Y = np.meshgrid(xvec, yvec)
    
    # Calculate Wigner function at specified coordinates
    W = wigner(psi, xvec, yvec)
    
    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig) 
    
    # plot surface
    ax.plot_surface(X, Y, W, rcount= 50, ccount= 50, cmap=cm.jet, alpha=.8)
    
    # Overlay contours onto plot
    ax.contour(X, Y, W, 10, zdir='x', offset=xrange[1])
    ax.contour(X, Y, W, 10, zdir='y', offset=yrange[1])
    ax.contour(X, Y, W, 20,zdir='z', offset=float(W.max() / 10))
    
    # Label Axes appropriately
    ax.set_xlabel(r'$\rm{Re}(\alpha) / x$')
    ax.set_ylabel(r'$\rm{Im}(\alpha) / p$')
    ax.set_zlabel(r'$W(\alpha)$')
    
    # Add title
    plt.title(title)
    return fig, ax

def my3dplot(X, Y, Z, title = 'Wigner Function'):
    x0 = X[0]
    y0 = Y[0]
    # Generate X and Y values from inputs
    X, Y = np.meshgrid(X, Y)
   
    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig) 
    
    # plot surface
    ax.plot_surface(X, Y, Z, rcount= 50, ccount= 50, cmap=cm.jet, alpha=.8)
    
    # Overlay contours onto plot
    ax.contour(X, Y, Z, 10, zdir='x', offset=x0)
    ax.contour(X, Y, Z, 10, zdir='y', offset=y0)
    ax.contour(X, Y, Z, 20,zdir='z', offset=0)
    
    # Label Axes appropriately
    ax.set_xlabel(r'$\rm{Re}(\alpha) / x$')
    ax.set_ylabel(r'$\rm{Im}(\alpha) / p$')
    ax.set_zlabel(r'$W(\alpha)$')
    
    # Add title
    plt.title(title)
    return fig, ax

def simps2d(Xvec, Yvec, Z):
    """
    Function to return value of the 2d simpson integration over a sample of values
    :Xvec: first axis of values at which function is evaluated
    :Yvec: second axis of values at which function is evaluated
    :Z: Function values, first indice being X and second Y
    """
    return scipy.integrate.simps(scipy.integrate.simps(Z, Xvec), Yvec)


def catstate(alpha, phi, theta, N):
    coh1 = np.cos(phi) * coherent(N, alpha)
    coh2 = np.sin(phi) * cmath.rect(1,theta) * coherent(N, -alpha)
    K = 1 + np.sin(2 * phi) * np.cos(theta) * np.exp(-2 * alpha * np.conj(alpha))
    norm = 1/np.sqrt(K)
    return norm * (coh1 + coh2)


def cubic(gamma, sqzf, N):
    x = position(N)
    cubeop = (1j * gamma * (x ** 3)).expm()
    sqop = squeeze(N, sqzf)
    
    cubic = cubeop * sqop * basis(N,0)
    
    return cubic


def innercubic(gamma, sqzf, N):
    exponent = (create(N) ** 3) + (destroy(N) ** 3)
    
    cubeop = (1j * gamma * exponent).expm()
    sqop = squeeze(N, sqzf)
    
    icubic = cubeop * sqop * basis(N,0)
    
    return icubic

def superposition(coeff, N):
    # TODO
    pass