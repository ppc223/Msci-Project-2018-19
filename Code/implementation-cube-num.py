import multiprocessing as mp
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
datafile = 'cubicnumerical.csv'
if not os.path.isfile(datafile):
    initoutput(datafile, columns)
    print('File Created')
else:
    moved = False
    i = 1
    while not moved:
        i = i + 1
        oldfile = datafile + '.old' + str(i)

        if not os.path.isfile(oldfile):
            print('Moved old file to to ' + oldfile)
            os.rename(datafile, oldfile)
            moved = True

    initoutput(datafile, columns)
    print('File Created')

# Initialise data lists, Prefix them with d to easily differentiate them.
dnorm = []
dWLN = []
dgamma = []
dr = []
dnmean = []
# dN = []
dxbound = []
# dxcount = []
dybound = []
# dycount = []
dtime1 = []
dtime2 = []

# Define the range parameters to iterate over and split into sections
# to export data regularly so as to not lose any if the program fails
splitcount = 8
range = np.concatenate((np.linspace(0.05,0.5,20), np.linspace(0.5,1,20)))

gammas = np.split(range, splitcount)
rs = range

percentstep = 1 / len(rs)

# Define some parameters to pass to WLN calculating function
xcount = ycount = 200

N = 150

# Iterate over all parameters defined
for ri, r in enumerate(rs):
    for i in np.arange(0, len(gammas)):

        # Set conditions here if only want some of the coordinates in the
        # defined ranges.
        #
        mask = np.ones(len(gammas[i]), dtype=bool)
        for k, gamma in enumerate(gammas[i]):
            theta = np.arctan(r / gamma)
            if gamma ** 2 + r ** 2 > (1/(np.cos(theta) - np.sin(theta))) ** 2:
                mask[k] = False

            if gamma < 0.5 and r < 0.5:
                mask[k] = False

        gammas[i] = gammas[i][mask]

        # Reset temporary arrays (used in parallel version to ensure correct
        # values are associated to correct outputs)
        states = []
        tempgamma = []
        tempnmean = []

        # Cacluate the states for each input parameter
        results1 = [[cubic(gamma, r, N), gamma] for gamma in gammas[i]]

        # Unpack the results
        for result in results1:
            states.append(result[0][0])
            tempnmean.append(result[0][1])
            tempgamma.append(result[1])

        # Calcuate the Wigner Logarithmic Negativity (Mana) for each state along
        # with the Wigner functions norm and the boundaries calculated by the
        # automatic boundary finding algorithm

        # TODO: Make it take last calculated boundaries as input until r
        # changes
        results2 = [[wln(state, 1e-5, xcount, ycount,
            initx = [-8,8], inity = [-8,8], incre = 1, maxdepth = 300),
            tempgamma[k], tempnmean[k]] for k, state in enumerate(states)]

        # Unpack the WLN results
        for result in results2:
            dWLN.append(result[0][0])
            dnorm.append(result[0][1])
            dtime1.append(result[0][2])
            dtime2.append(result[0][3])
            dxbound.append(result[0][4])
            dybound.append(result[0][5])
            dgamma.append(result[1])
            dnmean.append(result[2])
            dr.append(r)

        # Set once at start of program but may change between runs so still
        # to include these in data
        dxcount = [xcount,] * len(dgamma)
        dycount = [ycount,] * len(dgamma)
        dN = [N,] * len(dgamma)

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
            dgamma = []
            dr = []
            dnmean = []
            dxbound = []
            dxcount = []
            dybound = []
            dycount = []
            dtime1 = []
            dtime2 = []
            # Add some console output to see how the program is progressing
            # and if changes need to be made.
            print('exported data, progress = {:.4f} % in {:.4f} s'.format(
                (ri / len(rs) + (i * percentstep) / len(gammas)) * 100,
                time.time() - programstarttime))

print('Finished in {} s'.format(time.time() - programstarttime))
