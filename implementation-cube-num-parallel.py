import multiprocessing as mp
import os

import pandas as pd
import numpy as np

from funcs import *

# Column Heads for data output
columns=['Wnorm', 'WLN', 'gamma', 'r', 'nmean', 'Ndim','xbound', 'xcount',
    'ybound', 'ycount', 'time1', 'time2']

# Name datafile for output and if not existing create and add column headings
datafile = 'cubicnumerical.csv'
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
dnmean = []
dN = []
dxbound = []
dxcount = []
dybound = []
dycount = []
dtime1 = []
dtime2 = []

splitcount = 5
gammas = np.split(np.linspace(0.05, 0.5, 20), splitcount)
rs = np.linspace(0.05,0.5, 20)

xcount = ycount = 400

N = 200

for r in rs:
    for i in range(0, len(gammas)):
        pool = mp.Pool()
        states = []
        tempgamma = []
        tempnmean = []
        results1 = [[pool.apply_async(cubic, (gamma, r, N),
            dict([])), gamma]for gamma in gammas[i]]

        pool.close()
        pool.join()

        for result in results1:
            states.append(result[0].get()[0])
            tempnmean.append(result[0].get()[1])
            tempgamma.append(result[1])

        pool = mp.Pool()
        results2 = [[pool.apply_async(wln, (state, 1e-5, xcount, ycount),
            dict([])), tempgamma[i], tempnmean[i]]
            for i, state in enumerate(states)]

        pool.close()
        pool.join()

        for result in results2:
            dWLN.append(result[0].get()[0])
            dnorm.append(result[0].get()[1])
            dtime1.append(result[0].get()[2])
            dtime2.append(result[0].get()[3])
            dxbound.append(result[0].get()[4])
            dybound.append(result[0].get()[5])
            dgamma.append(result[1])
            dnmean.append(result[2])
            dr.append(r)

        dxcount = [xcount,] * len(dgamma)
        dycount = [ycount,] * len(dgamma)
        dN = [N,] * len(dgamma)

        data = list(zip(dnorm, dWLN, dgamma, dr, dnmean, dN, dxbound, dxcount,
            dybound, dycount, dtime1, dtime2))
        #,dN, dtime1, dtime2))

        df = pd.DataFrame(data, columns=columns)
        #
        # columns=['Wnorm', 'WLN', 'gamma', 'r', 'nmean', 'Ndim','xbound', 'xcount',
        #     'ybound', 'ycount', 'time1', 'time2']
        with open(datafile, 'a') as f:
            df.to_csv(f, header=False, index=False)
            dnorm = []
            dWLN = []
            dgamma = []
            dr = []
            dnmean = []
            dN = []
            dxbound = []
            dxcount = []
            dybound = []
            dycount = []
            dtime1 = []
            dtime2 = []
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
