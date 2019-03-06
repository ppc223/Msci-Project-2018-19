import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from mpl_toolkits.mplot3d import Axes3D
from scipy.special import airy
from matplotlib import cm
from qutip import *

# Plotting Functions:

def myplotwigner(psi, xrange = [0,3], yrange = [-20,20], step = 0.1,
        title = 'A Wigner Function', fineness = 50):
    """
    Function for plotting the Wiger function of a state which gives more control
    over the appearance of the graph than the built in qutip plot_wigner
    function.
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
    ax.plot_surface(X, Y, W, rcount=fineness, ccount=fineness,
        cmap=cm.jet, alpha=.8)

    # Overlay contours onto plot
    # ax.contour(X, Y, W, 10, zdir='x', offset=xrange[1])
    # ax.contour(X, Y, W, 10, zdir='y', offset=yrange[1])
    # ax.contour(X, Y, W, 20,zdir='z', offset=float(W.max() / 10))

    # Label Axes appropriately
    ax.set_xlabel(r'$\rm{Re}(\alpha) \ / \ x$')
    ax.set_ylabel(r'$\rm{Im}(\alpha) \ / \ p$')
    ax.set_zlabel(r'$W_{\rho}(\alpha)$')

    # Add title
    plt.title(title)
    return fig, ax


def my3dplot(X, Y, Z, title, axeslabels = ['x','y','z'], fineness = 50,
        cmap = cm.jet):
    """
    Function to plot an easily readable 3d visual surface plot
    of a set of input points.
    """
    x0 = X[0]
    y0 = Y[0]
    # Generate X and Y values from inputs
    X, Y = np.meshgrid(X, Y)

    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)

    # plot surface
    ax.plot_surface(X, Y, Z, rcount= fineness, ccount= fineness, cmap=cm.jet, alpha=.8, linewidth=0)

    # Overlay contours onto plot
    # ax.contour(X, Y, Z, 10, zdir='x', offset=x0)
    # ax.contour(X, Y, Z, 10, zdir='y', offset=y0)
    # ax.contour(X, Y, Z, 20,zdir='z', offset=0)

    # Label Axes appropriately
    ax.set_xlabel(axeslabels[0])
    ax.set_ylabel(axeslabels[1])
    ax.set_zlabel(axeslabels[2])

    # Add title
    plt.title(title)

    return fig, ax


def my3dscatter(X, Y, Z, title, axeslabels = ['x','y','z']):
    """
    Function to plot an easily readable 3d visual scatter plot
    of a set of input points.
    """
    #x0 = X[0]
    #y0 = Y[0]
    # Generate X and Y values from inputs
    #X, Y = np.meshgrid(X, Y)

    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)

    # plot surface
    ax.scatter(X, Y, Z, c=Z, cmap=cm.jet)

    # Overlay contours onto plot
    # ax.contour(X, Y, Z, 10, zdir='x', offset=x0)
    # ax.contour(X, Y, Z, 10, zdir='y', offset=y0)
    # ax.contour(X, Y, Z, 20,zdir='z', offset=0)

    # Label Axes appropriately
    ax.set_xlabel(axeslabels[0])
    ax.set_ylabel(axeslabels[1])
    ax.set_zlabel(axeslabels[2])

    # Add title
    plt.title(title)

    return fig, ax


# State Generating Functions

def catstate(alpha, phi, theta, N):
    """
    Generate a 'Cat State' Qobj, defined as a superposition of coherent states.
    :alpha: Argument of the coherent states.
    :phi: Amplitude (angular) parameter.
    :theta: Relative phase.
    :N: Number of fock states in Hilbert space.
    """
    # Generate Coherent States
    coh1 = np.cos(phi) * coherent(N, alpha)
    coh2 = np.sin(phi) * cmath.rect(1,theta) * coherent(N, -alpha)

    # Calculate Normalisation Factor
    K = 1 + np.sin(2 * phi) * np.cos(theta) * np.exp(-2 * alpha * np.conj(alpha))
    norm = 1/np.sqrt(K)

    # Calculate state and its expectation value for the number operator
    output = norm * (coh1 + coh2)
    nmean = expect(num(N), output)

    return output, nmean


def cubic(gamma, sqzf, N):
    """
    Generate a cubic phase state using a truncated fock space.
    :gamma: 'Cubicity' parameter.
    :sqzf: Squeezing factor on the state.
    :N: Number of fock states in Hilbert space.
    """
    # Define position and momentum operators
    x = position(N)
    p = momentum(N)

    # Define 'Cubic' and Squeezing operators in terms of x and p
    cubeop = (1j * gamma * (x ** 3)).expm()
    sqop = (-(1j * sqzf / 2) * (x * p + p * x)).expm()

    # Calculate state and its expectation value for the number operator
    cubic = cubeop * sqop * basis(N,0)
    nmean = expect(num(N), cubic)

    return cubic, nmean


def innercubic(gamma, sqzf, N):
    """
    Generate an 'Inner' cubic phase state, similar to the cubic phase state but
    with modal operators cubed in the 'Cubic operator' term rather than the
    position.
    :gamma: 'Cubicity' parameter.
    :sqzf: Squeezing factor on the state.
    :N: Number of fock states in Hilbert space.
    """
    # Define the position and momentum operators
    x = position(N)
    p = momentum(N)

    # Define the cubic operator exponent
    exponent = (create(N) ** 3) + (destroy(N) ** 3)

    # Define 'Cubic' and Squeezing operators
    cubeop = (1j * gamma * exponent).expm()
    sqop = (-(1j * sqzf / 2) * (x * p + p * x)).expm()

    # Calculate state and its expectation value for the number operator
    cubic = cubeop * sqop * basis(N,0)
    nmean = expect(num(N), cubic)

    return cubic, nmean


def wigcubic(X, P, gamma, r):
    """
    Calculate the value of the Wigner function for the cubic phase state from
    the expression given in arXiv:1809.05266.
    :X: Position coordinate (Real Coordinate).
    :P: Momentum coordinate (Imaginary Coordinate).
    :gamma: 'Cubicity' parameter.
    :r: Squeezing factor on the state.
    """
    # Create Grid from input X and P values
    x,p = np.meshgrid(X,P)

    # Calculate Normalisation factor
    N = np.exp(1 / (54 * (gamma ** 2) * np.exp(6 * r))) / (np.sqrt(np.pi) * np.exp(r)) * np.cbrt(4 / (3 * abs(gamma)))

    # Calcutate the argument of the Airy function at x and p
    c= np.cbrt(4 / (3 * gamma))
    airyarg = c * (3 * gamma * (x ** 2) - p + 1/(12 * gamma * np.exp(4 * r)))

    # Calculate the Wigner function
    Wcube = N * np.exp(-p/ (3 * gamma * np.exp(2 * r))) * airy(airyarg)[0]
    return Wcube


def superposition(coeff):
    """
    Generate a Superposition of Fock states.
    :coeff: List of coefficients, (preferably normalised), number of
    coefficents determines number of fock states in the Hilbert space.
    """
    # Calculate size of space
    N = len(coeff) + 1
    state = coeff[0] * basis(N,0)

    # Generate Superposition
    for i, x in enumerate(coeff[1:]):
        state = state + x * fock(N,i)

    # Calculate expectation value for the number operator
    expect(num(N), state)

    return state, nmean

# Calculation Functions:

def boundfindwig(state, tol, initx = [-3, 3], inity = [-3, 3], incre = 0.5,
        maxdepth = 30):
    """
    Rudimentary function for automatically finding bounds for a states wigner
    functionby looking at the value of the function along the edges of a grid
    and extendingthe grid until the values at the edges become below a
    tolerance.
    :state: The Qobj State vector/density matrix of a state
    :tol: Value considered negligible
    :initx/y: Inital x and y bounds
    :incre: Amount to change bounds by when point above tol found, also
    serves as spacing between points calculated along edges
    :maxdepth: maximum times to increase bounds before determining to stop
    """
    # Create x and p vectors
    xvec = np.arange(initx[0], initx[1], step = incre)
    yvec = np.arange(inity[0], inity[1], step = incre)

    # Initialise depth parameters
    d1 = 0
    d2 = 0

    # Need to re-evaluate if either vector is altered
    changed = True
    while changed:
        changed = False

        # if d1 > maxdepth:
        #     raise StopIteration('Max Depth reached, try increasing N or max depth')
        #
        # if d2 > maxdepth:
        #     raise StopIteration('Max Depth reached, try increasing N or max depth')

        if (np.abs(wigner(state, xvec, yvec[0])) > tol).any():
            yvec = np.insert(yvec, 0, yvec[0] - incre)
            d1 = d1 + 1
            changed = True

        if (np.abs(wigner(state, xvec, yvec[-1])) > tol).any():
            yvec = np.append(yvec, yvec[-1] + incre)
            d1 = d1 + 1
            changed = True


        if (np.abs(wigner(state, xvec[0], yvec)) > tol).any():
            xvec = np.insert(xvec, 0, xvec[0] - incre)
            d2 = d2 + 1
            changed = True

        if (np.abs(wigner(state, xvec[-1], yvec)) > tol).any():
            xvec = np.append(xvec, xvec[-1] + incre)
            d2 = d2 + 1
            changed = True


    return [xvec[0], xvec[-1] + incre], [yvec[0], yvec[-1] + incre]


def boundfindwigana(gamma, r, tol, initx = [-3, 3], inity = [-3, 3],
        incre = 0.5, maxdepth = 30 ):
    """
    Rudimentary function for automatically finding bounds for the cubic phase
    state using the analytic function defined by 'wigcubic' by looking at the
    value of the function along the edges of a grid and extendingthe grid until
    the values at the edges become below a tolerance.
    :tol: Value considered negligible.
    :gamma: 'Cubicity' parameter.
    :r: Squeezing factor on the state.
    :initx/y: Inital x and y bounds.
    :incre: Amount to change bounds by when point above tol found, also
    serves as spacing between points calculated along edges.
    :maxdepth: maximum times to increase bounds before determining to stop.
    """
    # Create x and p vectors
    xvec = np.arange(initx[0], initx[1] + incre, step = incre)
    yvec = np.arange(inity[0], inity[1] + incre, step = incre)

    # Initialise depth parameters
    d1 = 0
    d2 = 0

    # Need to re-evaluate if either vector is altered
    changed = True
    while changed:
        changed = False

        # if d1 > maxdepth:
        #     raise StopIteration('Max Depth reached, try increasing N or max depth')
        #
        # if d2 > maxdepth:
        #     raise StopIteration('Max Depth reached, try increasing N or max depth')

        if (np.abs(wigcubic(xvec, yvec[0], gamma, r)) > tol).any():
            yvec = np.insert(yvec, 0, yvec[0] - incre)
            d1 = d1 + 1
            changed = True

        if (np.abs(wigcubic(xvec, yvec[-1], gamma, r)) > tol).any():
            yvec = np.append(yvec, yvec[-1] + incre)
            d1 = d1 + 1
            changed = True

        if (np.abs(wigcubic(xvec[0], yvec, gamma, r)) > tol).any():
            xvec = np.insert(xvec, 0, xvec[0] - incre)
            d2 = d2 + 1
            changed = True

        if (np.abs(wigcubic(xvec[-1], yvec, gamma, r)) > tol).any():
            xvec = np.append(xvec, xvec[-1] + incre)
            d2 = d2 + 1
            changed = True

    return [xvec[0], xvec[-1] + incre], [yvec[0], yvec[-1] + incre]


def simps2d(Xvec, Yvec, Z):
    """
    Calculate the value of the 2d simpson integration over a sample of values.
    :Xvec: First axis of values at which function is evaluated.
    :Yvec: Second axis of values at which function is evaluated.
    :Z: Function values, first indice being X and second Y.
    """
    return scipy.integrate.simps(scipy.integrate.simps(Z, Xvec), Yvec)


def wln(state, tol, xcount=400, ycount=400, initx=[-3, 3], inity=[-3, 3],
        incre=0.5, maxdepth=80):
    """
    Calculate the normalisation and Wigner logarithmic negativity of
    a Qobj input state.
    :state: The Qobj State vector/density matrix of a state.
    :tol: Tolerance value for boundary finding.
    :x/ycount: Number of points to take along x/y axis.
    :initx/y: Inital x and y bounds.
    :incre: Increment value for boundary finding.
    :maxdepth: maximum times to increase bounds before determining to stop.
    """
    # Calculate boundaries for integration and record time for profiling
    boundtimestart = time.time()
    xbound, ybound = boundfindwig(state, tol, initx, inity, incre, maxdepth)
    boundtimeend = time.time()
    boundtime = boundtimeend - boundtimestart

    # Generate x and y vectors from calculated boundaries
    xvec = np.linspace(xbound[0], xbound[1], num = xcount)
    yvec = np.linspace(ybound[0], ybound[1], num = ycount)

    # record time to calculate the WLN
    calctimestart = time.time()
    # Calculate the Wigner function for the state
    W = wigner(state, xvec, yvec)
    # Calculate the normalisation of and WLN of the state
    Wnorm = simps2d(xvec, yvec, W)
    WLN = np.log2(simps2d(xvec, yvec, np.abs(W)))
    calctimeend = time.time()
    calctime = calctimeend - calctimestart

    return WLN, Wnorm, boundtime, calctime, xbound, ybound


def wlnanalytic(gamma, r, tol,  xcount=800, ycount=800, initx=[-3, 3],
        inity=[-3, 3], incre=0.5, maxdepth=80):
    """
    Calculate the normalisation and Wigner logarithmic negativity of the cubic
    phase state using the analytic expression calculated by 'wigcubic'.
    :gamma: 'Cubicity' parameter.
    :r: Squeezing factor on the state.
    :tol: Tolerance value for boundary finding.
    :x/ycount: Number of points to take along x/y axis.
    :initx/y: Inital x and y bounds.
    :incre: Increment value for boundary finding.
    :maxdepth: maximum times to increase bounds before determining to stop.
    """
    # Calculate boundaries for integration
    xbound, ybound = boundfindwigana(gamma, r, tol, initx, inity, incre,
        maxdepth)

    # Generate x and y vectors from calculated boundaries
    xvec = np.linspace(xbound[0], xbound[1], num = xcount)
    yvec = np.linspace(ybound[0], ybound[1], num = ycount)

    # Calculate the Wigner function for the state
    W = wigcubic(xvec, yvec, gamma, r)
    # Calculate the normalisation of and WLN of the state
    Wnorm = simps2d(xvec, yvec, W)
    WLN = np.log2(simps2d(xvec, yvec, np.abs(W)))
    return WLN, Wnorm, xbound, ybound

# Misc Functions:

def initoutput(datafile, columns):
    """
    Initialise an output csv file with given filename and
    column heads (as a list)
    """
    dfinit = pd.DataFrame(columns=columns)
    with open(datafile, 'a') as f:
        dfinit.to_csv(f, index=False)
