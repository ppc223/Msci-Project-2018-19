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
    'ybound', 'ycount']

# Name datafile for output and if not existing create and add column headings
datafile = 'cubicanalytic.csv'
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


# Initialise data lists
dnorm = []
dWLN = []
dgamma = []
dr = []
dxbound = []
dxcount = []
dybound = []
dycount = []
dN = []
dnmean = []

# Define the range parameters to iterate over and split into sections
# to export data regularly so as to not lose any if the program fails
splitcount = 10
range = np.linspace(0.05, 2, 100)
gammas = np.split(range, splitcount)
rs = range

percentstep = 1 / len(rs)

# Define some parameters to pass to WLN calculating function
xcount = ycount = 200

N = 200

# Iterate over all parameters defined
for ri, r in enumerate(rs):
    # Set inital x and y bounds for boundary finding and reset each time
    # we increment r
    initx = [-3, 3]
    inity = [-3, 3]
    for i in np.arange(0, len(gammas)):
        for k, gam in enumerate(gammas[i]):
            tWLN, tnorm, txbound, tybound = wlnanalytic(gam, r, 1e-4, xcount,
                ycount, initx, inity, 1, 300)

            tnmean = cubic(gam, r, N)[1]
            # Append data to associated lists
            dWLN.append(tWLN)
            dnorm.append(tnorm)
            dxbound.append(txbound)
            dybound.append(tybound)
            dgamma.append(gam)
            dr.append(r)
            dxcount.append(xcount)
            dycount.append(ycount)
            dN.append(N)
            dnmean.append(tnmean)

            # update initial x and y to those found in the last loop
            initx = txbound
            inity = tybound

        data = list(zip(dnorm, dWLN, dgamma, dr, dnmean, dN, dxbound, dxcount,
            dybound, dycount))

        df = pd.DataFrame(data, columns=columns)

        with open(datafile, 'a') as f:
            df.to_csv(f, header=False, index=False)
            dnorm = []
            dWLN = []
            dxbound = []
            dybound = []
            dgamma = []
            dr = []
            dxcount = []
            dycount = []
            dN = []
            dnmean = []
            # Add some console output to see how the program is progressing
            # and if changes need to be made.
            print('exported data, progress = {:.4f} % in {:.4f} s'.format(
                (ri / len(rs) + ((i + 1) * percentstep) / len(gammas)) * 100,
                time.time() - programstarttime))

print('Finished in {} s'.format(time.time() - programstarttime))
