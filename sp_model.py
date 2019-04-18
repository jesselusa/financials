# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16, 2019
@author: Jesse Lusa
"""

from computeFV import computeFV
from tabulate import tabulate
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(1)

# Imported S&P Performance
SPY = np.array([[2018, 2746.21, 2695.81, 2930.75, 2351.10, 2506.85, -6.24],\
                [2017, 2449.08, 2257.83, 2690.16, 2257.83, 2673.61, 19.42],\
                [2016, 2094.65, 2012.66, 2271.72, 1829.08, 2238.83, 9.54],\
                [2015, 2061.07, 2058.20, 2130.82, 1867.61, 2043.94, -0.73],\
                [2014, 1931.38, 1831.98, 2090.57, 1741.89, 2058.90, 11.39],\
                [2013, 1643.80, 1462.42, 1848.36, 1457.15, 1848.36, 29.60],\
                [2012, 1379.61, 1277.06, 1465.77, 1277.06, 1426.19, 13.41],\
                [2011, 1267.64, 1271.87, 1363.61, 1099.23, 1257.60, 0.00],\
                [2010, 1139.97, 1132.99, 1259.78, 1022.58, 1257.64, 12.78],\
                [2009, 948.05 , 931.80, 1127.78, 676.53, 1115.10, 23.45],\
                [2008, 1220.04, 1447.16, 1447.16, 752.44, 903.25, -38.49],\
                [2007, 1477.18, 1416.60, 1565.15, 1374.12, 1468.36, 3.53],\
                [2006, 1310.46, 1268.80, 1427.09, 1223.69, 1418.30, 13.62],\
                [2005, 1207.23, 1202.08, 1272.74, 1137.50, 1248.29, 3.00],\
                [2004, 1130.65, 1108.48, 1213.55, 1063.23, 1211.92, 8.99],\
                [2003, 965.23 , 909.03, 1111.92, 800.73, 1111.92, 26.38],\
                [2002, 993.93 , 1154.67, 1172.51, 776.76, 879.82, -23.37],\
                [2001, 1192.57, 1283.27, 1373.73, 965.80, 1148.08, -13.04],\
                [2000, 1427.22, 1455.22, 1527.46, 1264.74, 1320.28, -10.14],\
                [1999, 1327.33, 1228.10, 1469.25, 1212.19, 1469.25, 19.53],\
                [1998, 1085.50, 975.04, 1241.81, 927.69, 1229.23, 26.67],\
                [1997, 873.43 , 737.01, 983.79, 737.01, 970.43, 31.01],\
                [1996, 670.49 , 620.73, 757.03, 598.48, 740.74, 20.26],\
                [1995, 541.72 , 459.11, 621.69, 459.11, 615.93, 34.11],\
                [1994, 460.42 , 465.44, 482.00, 438.92, 459.27, -1.54],\
                [1993, 451.61 , 435.38, 470.94, 429.05, 466.45, 7.06],\
                [1992, 415.75 , 417.26, 441.28, 394.50, 435.71, 4.46],\
                [1991, 376.19 , 326.45, 417.09, 311.49, 417.09, 26.31],\
                [1990, 334.63 , 359.69, 368.95, 295.46, 330.22, -6.56],\
                [1989, 323.05 , 275.31, 359.80, 275.31, 353.40, 27.25]])

roiSPY = SPY[:, -1] / 100 
P = 1
for roi in roiSPY:
    P = P * (1 + roi)
avg_return = np.power(P, 1 / len(roiSPY)) - 1

# Fit Normal Distribution to ROI data
muSPY, stdSPY = norm.fit(roiSPY)
#plt.hist(roiSPY, bins=15, density=True)
#xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, 100)
#y = norm.pdf(x, muSPY, stdSPY)
#plt.plot(x, y)
#plt.close()

# Variables
investHorizon = 12 * 40 # months
monthlyContr = 100.0 # dollars
compoundFreq = 12 # (1) montlhly, (12) annually
nTrials = int(100000)

# Future-Value Model
totContr = investHorizon * monthlyContr
roiStocks = np.sort(roiSPY)

n = 12
t = int(np.floor(investHorizon / 12))
PMT = monthlyContr

FV = np.zeros((t, nTrials))
returns = np.zeros(np.shape(FV))
for nIdx in range(nTrials):
    P = 0.0
    tFV = np.zeros((t, 1))
    tReturns = np.zeros(np.shape(tFV))
    for tIdx in range(t):
        r = np.random.normal(muSPY, stdSPY)
        if r !=0.0:
            tmpFV = computeFV(P, PMT, compoundFreq, r, 1)
        else:
            tmpFV = tmpFV
        P = tmpFV
        tFV[tIdx, 0] = P
        tReturns[tIdx, 0] = r
    FV[:, nIdx] = tFV[:, 0]
    returns[:, nIdx] = tReturns[:, 0]
    
# Analysis
capital_expnd = np.linspace(monthlyContr * 12, totContr, t)
capital_expnd = np.array([capital_expnd,]*nTrials).transpose()
total_roi = (FV - capital_expnd) / capital_expnd

years = np.linspace(1, t, t)
years_of_interest = np.array([20, 30, 35, 40])
stats_arr = np.zeros((len(years_of_interest), 7))
stats_arr[:, 0] = np.transpose(years_of_interest)
arr_headers = np.array(['Investment_Horizon', 'Capital Investment', 'Mean', 'Std',\
                        '10th Percentile', '50th Percentile', '70th Percentile',])

# Plotting              
fig, ax = plt.subplots(2, 1, figsize=(16, 10)) #figsize = (16, 10))
for yIdx in range(len(years_of_interest)):
    year = years_of_interest[yIdx]
    idx = np.where(years == year)[0][0]
    FV_year = FV[idx, :]
    stats_arr[yIdx, 1:3] = norm.fit(FV_year)
    stats_arr[yIdx, 3] = monthlyContr * investHorizon
    stats_arr[yIdx, 4:7] = np.array([np.percentile(FV_year, 10),\
                                     np.percentile(FV_year, 50),\
                                     np.percentile(FV_year, 75)])
    ax[0].hist(FV_year, bins=int(nTrials/100), range=[0, 1.5e4 * monthlyContr], density=True, cumulative=-1,
            histtype = 'step', label='{0:d} Years'.format(year))
    roi_year =  total_roi[idx, :]
    ax[1].hist(roi_year, bins=int(nTrials/100), range=[0, 65], density=True, cumulative=-1,
            histtype = 'step', label='{0:d} Years'.format(year))

print('\n')
print(tabulate(stats_arr, headers=arr_headers))
print('\n')

ax[0].set_yticks(np.arange(0, 1.1, 0.10))
ax[0].axis(xmin=0, ymin=0, ymax=1)
ax[0].grid(True)
ax[0].legend(loc='right')
ax[0].set_title('Cumulative Density Function of Future Values, ${0:0.0f}/Month Deposit'.format(monthlyContr))
ax[0].set_xlabel('Future Value (USD)')
ax[0].set_ylabel('Probability of Occurrence')

ax[1].set_yticks(np.arange(0, 1.1, 0.10))
ax[1].set_xticks(np.arange(0, 75, 5))
ax[1].axis(xmin=0, ymin=0, ymax=1)
ax[1].grid(True)
ax[1].legend(loc='right')
ax[1].set_title('Cumulative Density Function of ROI, ${0:0.0f}/Month Deposit'.format(monthlyContr))
ax[1].set_xlabel('Return on Investment')
ax[1].set_ylabel('Probability of Occurrence')