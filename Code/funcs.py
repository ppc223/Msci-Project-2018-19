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

def myplotwigner(psi, xrange = [-3,3], yrange = [-3,3], step = 0.1,
        title='', rccount = 50, fill = True, cont = False):
    """
    Function for plotting the Wiger function of a state which gives more control
    over the appearance of the graph than the built in qutip plot_wigner
    function.

    Parameters
    ----------
    psi : Qobj
        The state vector or density operator to plot the Wigner function of.
    xrange, yrange : array-like(2)
        The boundaries of the quantum phase space in which to plot the Wigner
        function
    step : float
        The grid spaceing to calculate the Wigner function at
    title : string
        The title to attach to the plot
    rccount : int
        Maximum number of samples used in both directions.  If the input
        data is larger, it will be downsampled (by slicing) to these
        numbers of points.
    cmap : Matplotlib Colormap object
        The colormap to apply to the points.
    cont : bool
        Whether or not to plot contours on all the axes.
    fill : bool
        Whether or not to fill the plot background.
    grids : bool
        Whether ot not to render gridlines

    Returns
    -------
    figure : `~matplotlib.figure.Figure`
        The `.Figure` instance returned will also be passed to new_figure_manager
        in the backends, which allows to hook custom `.Figure` classes into the
        pyplot interface. Additional kwargs will be passed to the `.Figure`
        init function.
    axes : `~.axes.Axes` (or a subclass of `~.axes.Axes`)
        The returned axes class depends on the projection used. It is
        `~.axes.Axes` if rectilinear projection are used and
        `.projections.polar.PolarAxes` if polar projection
        are used.
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
    ax.plot_surface(X, Y, W, rcount=rccount, ccount=rccount,
        cmap=cm.jet, alpha=.8)

    # Overlay contours onto plot
    if cont:
        ax.contour(X, Y, W, 10, zdir='x', offset=xrange[1])
        ax.contour(X, Y, W, 10, zdir='y', offset=yrange[1])
        ax.contour(X, Y, W, 20, zdir='z', offset=float(W.max() / 10))

    # Label Axes appropriately
    ax.set_xlabel(r'$\rm{Re}(\alpha) \ / \ x$')
    ax.set_ylabel(r'$\rm{Im}(\alpha) \ / \ p$')
    ax.set_zlabel(r'$W_{\rho}(\alpha)$')

    # Remove background grid
    ax.grid(False)

    # Remove background fill
    if not fill:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # Make pane around each axes black, adds a border to plot
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # Sets the pane borders to be closed squares, by default only 3 sides
    ax.xaxis.pane.set_closed(True)
    ax.yaxis.pane.set_closed(True)
    ax.zaxis.pane.set_closed(True)

    # Add title
    plt.title(title)

    return fig, ax


def my3dplot(X, Y, Z, title, axeslabels = ['x','y','z'], rccount = 50,
        cmap = cm.jet, cont=False, fill = True, grids = True):
    """
    Function to plot an easily readable 3d surface plot
    of a set of input points.

    Parameters
    ----------
    X, Y, Z : 2d arrays
        Data values.
    title : string
        The title to attach to the plot.
    axelabels : array-like(3), elements are strings
        The axes labels to attach to the plot.
    rccount : int
        Maximum number of samples used in both directions.  If the input
        data is larger, it will be downsampled (by slicing) to these
        numbers of points.
    cmap : Matplotlib Colormap object
        The colormap to apply to the points.
    cont : bool
        Whether or not to plot contours on all the axes.
    fill : bool
        Whether or not to fill the plot background.
    grids : bool
        Whether ot not to render gridlines

    Returns
    -------
    figure : `~matplotlib.figure.Figure`
        The `.Figure` instance returned will also be passed to new_figure_manager
        in the backends, which allows to hook custom `.Figure` classes into the
        pyplot interface. Additional kwargs will be passed to the `.Figure`
        init function.
    axes : `~.axes.Axes` (or a subclass of `~.axes.Axes`)
        The returned axes class depends on the projection used. It is
        `~.axes.Axes` if rectilinear projection are used and
        `.projections.polar.PolarAxes` if polar projection
        are used.
    """
    x0 = X[0]
    y0 = Y[0]
    # Generate X and Y values from inputs
    X, Y = np.meshgrid(X, Y)

    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)

    # plot surface
    ax.plot_surface(X, Y, Z, rcount= rccount, ccount= rccount, cmap=cm.jet, alpha=.8, linewidth=0)

    # Overlay contours onto plot
    if cont:
        ax.contour(X, Y, Z, 10, zdir='x', offset=x0)
        ax.contour(X, Y, Z, 10, zdir='y', offset=y0)
        ax.contour(X, Y, Z, 20, zdir='z', offset=0)

    # Label Axes appropriately
    ax.set_xlabel(axeslabels[0])
    ax.set_ylabel(axeslabels[1])
    ax.set_zlabel(axeslabels[2])

    # Remove background grid
    if not grids:
        ax.grid(False)

    # Remove background fill
    if not fill:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # Add title
    plt.title(title)

    return fig, ax


def my3dscatter(X, Y, Z, title, axeslabels = ['x','y','z'], cmap = cm.jet):
    """
    Function to plot an easily readable 3d scatter plot
    of a set of input points.

    Parameters
    ----------
    X, Y : array-like
        The coordinates of the values in Z.
    Z : array-like(N, M)
        The height values of the points.
    title : string
        The title to attach to the plot.
    axelabels : array-like(3), elements are strings
        The axes labels to attach to the plot.
    cmap : Matplotlib Colormap object
        The colormap to apply to the points.

    Returns
    -------
    figure : `~matplotlib.figure.Figure`
        The `.Figure` instance returned will also be passed to new_figure_manager
        in the backends, which allows to hook custom `.Figure` classes into the
        pyplot interface. Additional kwargs will be passed to the `.Figure`
        init function.
    axes : `~.axes.Axes` (or a subclass of `~.axes.Axes`)
        The returned axes class depends on the projection used. It is
        `~.axes.Axes` if rectilinear projection are used and
        `.projections.polar.PolarAxes` if polar projection
        are used.
    """
    #x0 = X[0]
    #y0 = Y[0]
    # Generate X and Y values from inputs
    #X, Y = np.meshgrid(X, Y)

    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)

    # plot surface
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)

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


def mycontourplot(X, Y, Z, title, axeslabels = ['q','p'],
        cmap = cm.jet, levels = 50):
    """
    Function to plot an easily readable contour plot
    of a set of input points.

    Parameters
    ----------
    X, Y : array-like
        The coordinates of the values in Z.
    Z : array-like(N, M)
        The height values over which the contour is drawn.
    title : string
        The title to attach to the plot
    axelabels : array-like(2), elements are strings
        The axes labels to attach to the plot
    cmap : Matplotlib Colormap object
        The colormap to apply to the contours

    Returns
    -------
    figure : `~matplotlib.figure.Figure`
        The `.Figure` instance returned will also be passed to new_figure_manager
        in the backends, which allows to hook custom `.Figure` classes into the
        pyplot interface. Additional kwargs will be passed to the `.Figure`
        init function.
    axes : `~.axes.Axes` (or a subclass of `~.axes.Axes`)
        The returned axes class depends on the projection used. It is
        `~.axes.Axes` if rectilinear projection are used and
        `.projections.polar.PolarAxes` if polar projection
        are used.
    """
    x0 = X[0]
    y0 = Y[0]
    # Generate X and Y values from inputs
    X, Y = np.meshgrid(X, Y)

    # Create Figure and Axes for the plot
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    # plot Contour
    cp = plt.contourf(X, Y, Z, levels = levels, cmap=cmap)
    plt.colorbar(cp)

    # Label Axes appropriately
    ax.set_xlabel(axeslabels[0])
    ax.set_ylabel(axeslabels[1])

    # Add title
    plt.title(title)

    return fig, ax


# State Generating Functions

def catstate(alpha, phi, theta, N):
    """
    Generate a 'Cat State' Qobj, defined as a superposition of coherent states.

    Parameters
    ----------
    alpha : float/complex
        Argument of the Displacement operator generating the coherent
        state.
    phi : float
        Amplitude (angular) parameter.
    theta : float
        Relative phase.
    N : int
        Number of fock states in Hilbert space.

    Returns
    -------
    cat : Qobj
        The requested cat state
    nmean : float
        The mean of the number operator over the state <n>
    """
    # Generate Coherent States
    coh1 = np.cos(phi) * coherent(N, alpha)
    coh2 = np.sin(phi) * cmath.rect(1,theta) * coherent(N, -alpha)

    # Calculate Normalisation Factor
    K = 1 + np.sin(2 * phi) * np.cos(theta) * np.exp(-2 * alpha * np.conj(alpha))
    norm = 1/np.sqrt(K)

    # Calculate state and its expectation value for the number operator
    cat = norm * (coh1 + coh2)
    nmean = expect(num(N), cat)

    return cat, nmean


def cubic(gamma, sqzf, N):
    """
    Generate a Cubic Phase state with Cubicity gamma and Squeezing sqzf.
    Constructed using a squeezing operator then the cubic phase gate on vacuum.

    Parameters
    ----------
    gamma : float/complex
        The Cubicity parameter for the cubic phase state
    sqzf : float/complex
        The squeezing parameter for the cubic phase state
    N : int
        Number of fock states in Hilbert space.

    Returns
    -------
    cubic : Qobj
        The requested cubic phase state
    nmean : float
        The mean of the number operator over the state <n>
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
    Generate a Triple Photon state with Cubicity gamma and Squeezing sqzf.
    Named innercubic as it was previously what the state was called internally.
    Constructed using a squeezing operator then the cubic phase gate on vacuum.

    Parameters
    ----------
    gamma : float/complex
        The triplicity parameter for the triple photon state
    sqzf : float/complex
        The squeezing parameter for the triple photon state
    N : int
        Number of fock states in Hilbert space.

    Returns
    -------
    cubic : Qobj
        The requested triple photon state
    nmean : float
        The mean of the number operator over the state <n>,
        this is calculated using the numeric approach in qutip by generating
        the state and then appliying the operator to it.
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
    triple = cubeop * sqop * basis(N,0)
    nmean = expect(num(N), triple)

    return triple, nmean


def wigcubic(X, P, gamma, r):
    """
    Calculate the value of the Wigner function for the cubic phase state from
    the semi-analytic expression given in arXiv:1809.05266 in terms of the
    Airy function.

    Parameters
    ----------
    X : array_like
        x-coordinates at which to calculate the Wigner function.
    P : array_like
        p-coordinates at which to calculate the Wigner function.
    gamma : float/complex
        The triplicity parameter for the triple photon state.
    sqzf : float/complex
        The squeezing parameter for the triple photonstate.

    Wcube : array
        Values representing the Wigner function calculated over the specified
        range [X,P].
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
    Parameters
    ----------
    coeff : array_like
        List of coefficients, (preferably normalised), number of coefficents
        determines number of fock states in the Hilbert space.

    Returns
    -------
    state : Qobj
        Qobj representing the requested superposition of number states.
    nmean : float
        <n> of the state, n being the number operator
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

def _boundfindwig_old(state, tol, initx = [-3, 3], inity = [-3, 3],
        incre = 0.5, maxdepth = 30):
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


def boundfindwig(state, tol, initx = [-3, 3], inity = [-3, 3], incre = 0.5):
    """
    Rudimentary function for automatically finding bounds for a states wigner
    functionby looking at the value of the function along the edges of a grid
    and extending the grid until the values at the edges become below a
    tolerance, or contracting the grid until at least one values is above the
    tolerance.

    Parameters
    ----------
    state : Qobj
        The Qobj State vector/density matrix of a state
    tol : number-like
        Value considered negligible
    initx/y : 2-d list
        Inital x and y bounds
    incre : number-like
        Amount to change bounds by when point above tol found, also
        serves as spacing between points calculated along edges

    Returns
    -------
    xbounds: 2-d list
        lower x boundary and upper x boundary in a list
    ybounds: 2-d list
        lower y boundary and upprt y boundary in a list
    """
    # Create q and p vectors
    xvec = np.arange(initx[0], initx[1], step = incre)
    yvec = np.arange(inity[0], inity[1], step = incre)

    changed = np.ones(4, bool)
    expanding = np.ones(4, bool)
    firstrun = True

    while changed.any():
        lastchanged = np.copy(changed)
        changed = np.zeros(4, bool)

        if lastchanged[0]: # Scanning along x at min y
            if expanding[0]:
                if (np.abs(wigner(state, xvec, yvec[0])) > tol).any():
                    yvec = np.insert(yvec, 0, yvec[0] - incre)
                    changed[0] = True
                elif firstrun:
                    expanding[0] = False
                    changed[0] = True
            # Scans along x at one increment above min y
            elif not (np.abs(wigner(state, xvec, yvec[1])) > tol).any():
                yvec = np.delete(yvec, 0)
                changed[0] = True

        if lastchanged[1]: # Scanning along x at max y
            if expanding[1]:
                if (np.abs(wigner(state, xvec, yvec[-1])) > tol).any():
                    yvec = np.append(yvec, yvec[-1] + incre)
                    changed[1] = True
                elif firstrun:
                    expanding[1] = False
                    changed[1] = True
            elif not (np.abs(wigner(state, xvec, yvec[-2])) > tol).any():
                yvec = np.delete(yvec, -1)
                changed[1] = True

        if lastchanged[2]: # Scanning along y at min x
            if expanding[2]:
                if (np.abs(wigner(state, xvec[0], yvec)) > tol).any():
                    xvec = np.insert(xvec, 0, xvec[0] - incre)
                    changed[2] = True
                elif firstrun:
                    expanding[2] = False
                    changed[2] = True
            elif not (np.abs(wigner(state, xvec[1], yvec)) > tol).any():
                xvec = np.delete(xvec, 0)
                changed[2] = True


        if lastchanged[3]: # Scanning along y at max x
            if expanding[3]:
                if (np.abs(wigner(state, xvec[-1], yvec)) > tol).any():
                    xvec = np.append(xvec, xvec[-1] + incre)
                    changed[3] = True
                elif firstrun:
                    expanding[3] = False
                    changed[3] = True
            elif not (np.abs(wigner(state, xvec[-2], yvec)) > tol).any():
                xvec = np.delete(xvec, -1)
                changed[3] = True

        firstrun = False

    return [xvec[0], xvec[-1]], [yvec[0], yvec[-1]]


def boundfindwigana(gamma, r, tol, initx = [-3, 3], inity = [-3, 3],
        incre = 0.5):
    """
    Rudimentary function for automatically finding bounds for the semi-analytic
    form of the Wigner function for the cubic phase state by looking at the
    value of the function along the edges of a grid and extending the grid
    until the values at the edges become below a tolerance.
    Doesn't have the contracting functionality of the non-analytic version.

    Parameters
    ----------
    gamma : float/complex
        The Cubicity parameter for the cubic phase state
    sqzf : float/complex
        The squeezing parameter for the cubic phase state
    state : Qobj
        The Qobj State vector/density matrix of a state
    tol : number-like
        Value considered negligible
    initx/y : 2-d list
        Inital x and y bounds
    incre : number-like
        Amount to change bounds by when point above tol found, also
        serves as spacing between points calculated along edges

    Returns
    -------
    xbounds: 2-d list
        lower x boundary and upper x boundary in a list
    ybounds: 2-d list
        lower y boundary and upprt y boundary in a list
    """
    # Create x and p vectors
    xvec = np.arange(initx[0], initx[1] + incre, step = incre)
    yvec = np.arange(inity[0], inity[1] + incre, step = incre)

    # Need to re-evaluate if either vector is altered
    changed = True
    while changed:
        changed = False
        if (np.abs(wigcubic(xvec, yvec[0], gamma, r)) > tol).any():
            yvec = np.insert(yvec, 0, yvec[0] - incre)
            changed = True

        if (np.abs(wigcubic(xvec, yvec[-1], gamma, r)) > tol).any():
            yvec = np.append(yvec, yvec[-1] + incre)
            changed = True

        if (np.abs(wigcubic(xvec[0], yvec, gamma, r)) > tol).any():
            xvec = np.insert(xvec, 0, xvec[0] - incre)
            changed = True

        if (np.abs(wigcubic(xvec[-1], yvec, gamma, r)) > tol).any():
            xvec = np.append(xvec, xvec[-1] + incre)
            changed = True

    return [xvec[0], xvec[-1]], [yvec[0], yvec[-1]]


def simps2d(Xvec, Yvec, Z):
    """
    Calculate the value of the 2d simpson integration over a sample of values.

    Parameters
    ----------
    Xvec : array_like
        First axis of values at which function is evaluated.
    Yvec: array_like
        Second axis of values at which function is evaluated.
    Z: array_like
        Function values, first indice being X and second Y.

    Returns
    -------
    The value of the integration.
    """
    return scipy.integrate.simps(scipy.integrate.simps(Z, Xvec), Yvec)


def wln(state, tol, xcount=400, ycount=400, initx=[-3, 3], inity=[-3, 3],
        incre=0.5):
    """
    Calculate the normalisation of the Wigner function and the Wigner
    logarithmic negativity of a Qobj input state.

    Parameters
    ----------
    state: Qobj
        The Qobj State vector/density matrix of a state.
    tol: float
        Tolerance value for boundary finding algorithm - boundfindwig.
    x/ycount: int
        Number of points to take along x/y axis.
    initx/y: array_like(2)
        Inital x and y bounds.
    incre: float
        Increment value for boundary finding.
    maxdepth:
        maximum times to increase bounds before determining to stop.

    Returns
    -------
    WLN : float
        The value of the Wigner logarithmic negativity of the input state
    Wnorm : float
        The normalisation of the Wigner function calculated
    bound/calctime : float
        The time taken to calculate the boundaries of the phase space and then to calculate the Wigner function at the grid of points within
    x/ybound : array_like(2)
        The calculated x and y bounds of the phase space
    """
    # Calculate boundaries for integration and record time for profiling
    boundtimestart = time.time()
    xbound, ybound = boundfindwig(state, tol, initx, inity, incre)
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
        inity=[-3, 3], incre=0.5):
    """
    Calculate the normalisation of the Wigner function and the Wigner
    logarithmic negativity of the cubic phase state using the semi-analytic
    solution given by wigcubic.

    Parameters
    ----------
    gamma : float/complex
        The Cubicity parameter for the cubic phase state
    sqzf : float/complex
        The squeezing parameter for the cubic phase state
    tol: float
        Tolerance value for boundary finding algorithm - boundfindwig.
    x/ycount: int
        Number of points to take along x/y axis.
    initx/y: array_like(2)
        Inital x and y bounds.
    incre: float
        Increment value for boundary finding.
    maxdepth:
        maximum times to increase bounds before determining to stop.

    Returns
    -------
    WLN : float
        The value of the Wigner logarithmic negativity of the input state
    Wnorm : float
        The normalisation of the Wigner function calculated
    x/ybound : array_like(2)
        The calculated x and y bounds of the phase space
    """
    # Calculate boundaries for integration
    xbound, ybound = boundfindwigana(gamma, r, tol, initx, inity, incre)

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
    Initialise an output csv file with given filename and column heads (as a list)

    Parameters
    ----------
    datafile : string
        The name of the csv file, with csv extension.
    columns : array_like of strings
        The titles of the columns for the datafile.
    """
    dfinit = pd.DataFrame(columns=columns)
    with open(datafile, 'a') as f:
        dfinit.to_csv(f, index=False)
