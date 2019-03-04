import multiprocessing as mp
import warnings
import time
import os

import pandas as pd
import numpy as np
from funcs import *

# Record time the program starts to see total runtime
programstarttime = time.time()

# Column Heads for data output
columns=['Wnorm', 'WLN', 'gamma', 'r', 'nmean', 'Ndim','xbound', 'xcount',
    'ybound', 'ycount', 'time1', 'time2']

# Name datafile for output and if not existing create and add column headings
# if it does exist move the current file to a new file to avoid contaminating
# either with bad data.
datafile = 'inner_cubic.csv'
if not os.path.isfile(datafile):
    initoutput(datafile, columns)
    print('File Created')
else:
    moved = False
    i = 1
    while not moved:
        oldfile = datafile + '.old' + str(i)

        if not os.path.isfile(oldfile):
            print('Moved old file to to ' + oldfile)
            os.rename(datafile, oldfile)
            moved = True
        i = i + 1

    initoutput(datafile, columns)
    print('File Created')

# Initialise data lists, Prefix them with d to easily differentiate them.
dnorm = []
dWLN = []
dtime1 = []
dtime2 = []
dxbound = []
dybound = []
dnmean = []
dgamma = []
dr = []
dxcount = []
dycount = []
dN = []

# Define the range parameters to iterate over and split into sections
# to export data regularly so as to not lose any if the program fails
splitcount = 10
range = np.linspace(0, 1, 60)
gammas = np.split(range * 0.3, splitcount)
rs = range * 1.7

percentstep = 1 / len(rs)

# Define some parameters to pass to WLN calculating function
xcount = ycount = 200

N = 150

# Iterate over all parameters defined
for ri, r in enumerate(rs):
    # Set inital x and y bounds for boundary finding and reset each time
    # we increment r
    initx = [-3, 3]
    inity = [-3, 3]
    for i in np.arange(0, len(gammas)):

        # Set conditions here if only want some of the coordinates in the
        # defined ranges.
        #
        # mask = np.ones(len(gammas[i]), dtype=bool)
        # for k, gamma in enumerate(gammas[i]):
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings('error')
        #         try:
        #             theta = np.arctan(r / gamma)
        #         except RuntimeWarning:
        #             theta = np.pi / 2
        #     # Want results in half of the unit square described by right
        #     # angled triangle with right angle cusped by the axes at 0, 0
        #     if gamma ** 2 + r ** 2 > (1/(np.cos(theta) + np.sin(theta))) ** 2:
        #         mask[k] = False
        #
        # gammas[i] = gammas[i][mask]

        # Reset (and initialise) list of states from previous loops
        states = []

        # Cacluate the states for each input parameter
        for k, gamma in enumerate(gammas[i]):
            [tstates, tnmean], tgamma = innercubic(gamma, r, N), gamma
            # Store states and associated nmean and gamma values in a list,
            # more useful if trying to parallelize the code
            states.append([tstates,tnmean,tgamma])

        # Calcuate the Wigner Logarithmic Negativity (Mana) for each state along
        # with the Wigner functions norm and the boundaries calculated by the
        # automatic boundary finding algorithm
        for k, state in enumerate(states):
            [tWLN, tnorm, ttime1, ttime2, txbound, tybound], tnmean, tgamma = [
                wln(state[0], 1e-4, xcount, ycount, initx, inity, 1, 300),
                state[1], state[2]]
            # Append data to associated lists
            dWLN.append(tWLN)
            dnorm.append(tnorm)
            dtime1.append(ttime1)
            dtime2.append(ttime2)
            dxbound.append(txbound)
            dybound.append(tybound)
            dnmean.append(tnmean)
            dgamma.append(tgamma)
            dr.append(r)
            dxcount.append(xcount)
            dycount.append(ycount)
            dN.append(N)

            # update initial x and y to those found in the last loop
            initx = txbound
            inity = tybound

        # Zip lists together before outputing them to file
        data = list(zip(dnorm, dWLN, dgamma, dr, dnmean, dN, dxbound, dxcount,
            dybound, dycount, dtime1, dtime2))

        # Create Pandas dataframe from the data
        df = pd.DataFrame(data, columns=columns)

        # Output the data to file and reset the data lists
        with open(datafile, 'a') as f:
            df.to_csv(f, header=False, index=False)
            dnorm = []
            dWLN = []
            dtime1 = []
            dtime2 = []
            dxbound = []
            dybound = []
            dnmean = []
            dgamma = []
            dr = []
            dxcount = []
            dycount = []
            dN = []
            # Add some console output to see how the program is progressing
            # and if changes need to be made.
            print('exported data, progress = {:.4f} % in {:.4f} s'.format(
                (ri / len(rs) + ((i + 1) * percentstep) / len(gammas)) * 100,
                time.time() - programstarttime))

print('Finished in {} s'.format(time.time() - programstarttime))
