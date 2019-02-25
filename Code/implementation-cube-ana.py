import multiprocessing as mp
import os

import pandas as pd
import numpy as np

from funcs import *

# Column Heads for data output
columns=['Wnorm', 'WLN', 'gamma', 'r', 'xbound', 'xcount',
    'ybound', 'ycount']

# Name datafile for output and if not existing create and add column headings
datafile = 'cubicanalytic.csv'
if not os.path.isfile(datafile):
    dfinit = pd.DataFrame(columns=columns)
    with open(datafile, 'a') as f:
        dfinit.to_csv(f, index=False)
    print('File Created')

# Initialise data lists
dnorm = []
dWLN = []
dgamma = []
dr = []
dxbound = []
# dxcount = []
dybound = []
# dycount = []
# dN = []
# dtime1 = []
# dtime2 = []

pool = mp.Pool(processes=8)

splitcount = 10
gammas = np.split(np.linspace(0.1, 0.5, 20), splitcount)
rs = np.linspace(0.1,0.5, 20)

xcount = ycount = 400

for r in rs:
    for i in range(0, len(gammas)):
        results = [pool.apply(wlnanalytic, (gamma, r, 1e-5, xcount, ycount),
            dict([])) for gamma in gammas[i]]

        dgamma = dgamma + list(gammas[i])
        dxcount = [xcount,] * len(dgamma)
        dycount = [ycount,] * len(dgamma)

        for result in results:
            dWLN.append(result[0])
            dnorm.append(result[1])
            dxbound.append(result[2])
            dybound.append(result[3])
            dr.append(r)

        data = list(zip(dnorm, dWLN, dgamma, dr, dxbound, dxcount,
            dybound, dycount))

        df = pd.DataFrame(data, columns=columns)

        with open(datafile, 'a') as f:
            df.to_csv(f, header=False, index=False)
            dnorm = []
            dWLN = []
            dgamma = []
            dr = []
            dxbound = []
            dxcount = []
            dybound = []
            dycount = []
            print('exported data')



# WLN, Wnorm, xbound, ybound = wlnanalytic(gamma, sqz, 1e-7)

# columns=['Wnorm', 'WLN', 'gamma', 'r', 'xbound', 'xcount',
#    'ybound', 'ycount', 'N', 'time1', 'time2']

# data = list(zip(dnorm, dWLN, dgamma, dr, dxbound, dxcount, dybound, dxcount)
# #,dN, dtime1, dtime2))
#
# df = pd.DataFrame(data, columns=columns)
#
# with open(datafile, 'a') as f:
#     df.to_csv(f, header=False, index=False)
#     written = True
