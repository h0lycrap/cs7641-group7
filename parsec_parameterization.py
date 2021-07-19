from __future__ import division
from math import sqrt, tan, pi
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims
import pandas as pd
from scipy.optimize import minimize
from time import time

# INPUT.csv must be placed in the same directory as the script

# Code copied and modified from https://github.com/dqsis/parsec-airfoils
def pcoef(
        xte,yte,rle,
        x_cre,y_cre,d2ydx2_cre,th_cre,
        surface):
    """evaluate the PARSEC coefficients"""

    # Initialize coefficients
    coef = np.zeros(6)

    # 1st coefficient depends on surface (pressure or suction)
    if surface.startswith('p'):
        coef[0] = -sqrt(2*rle)
    else:
        coef[0] = sqrt(2*rle)
 
    # Form system of equations
    A = np.array([
                 [xte**1.5, xte**2.5, xte**3.5, xte**4.5, xte**5.5],
                 [x_cre**1.5, x_cre**2.5, x_cre**3.5, x_cre**4.5, 
                  x_cre**5.5],
                 [1.5*sqrt(xte), 2.5*xte**1.5, 3.5*xte**2.5, 
                  4.5*xte**3.5, 5.5*xte**4.5],
                 [1.5*sqrt(x_cre), 2.5*x_cre**1.5, 3.5*x_cre**2.5, 
                  4.5*x_cre**3.5, 5.5*x_cre**4.5],
                 [0.75*(1/sqrt(x_cre)), 3.75*sqrt(x_cre), 8.75*x_cre**1.5, 
                  15.75*x_cre**2.5, 24.75*x_cre**3.5]
                 ]) 

    B = np.array([
                 [yte - coef[0]*sqrt(xte)],
                 [y_cre - coef[0]*sqrt(x_cre)],
                 [tan(th_cre*pi/180) - 0.5*coef[0]*(1/sqrt(xte))],
                 [-0.5*coef[0]*(1/sqrt(x_cre))],
                 [d2ydx2_cre + 0.25*coef[0]*x_cre**(-1.5)]
                 ])
    
    # Solve system of linear equations
    X = np.linalg.solve(A,B) 

    # Gather all coefficients
    coef[1:6] = X[0:5,0]

    # Return coefficients
    return coef


def ppoints(cf_pre, cf_suc, npts=44, xte=1.0):
    '''
    Takes PARSEC coefficients, number of points, and returns list of
    [x,y] coordinates starting at trailing edge pressure side.
    Assumes trailing edge x position is 1.0 if not specified.
    Returns 121 points if 'npts' keyword argument not specified.
    '''
    # Using cosine spacing to concentrate points near TE and LE,
    # see http://airfoiltools.com/airfoil/naca4digit
    # modfied from 121 to 44 to match the given data
    xpts = (1 - np.cos(np.linspace(0, 1, int(np.ceil(npts/2)))*np.pi)) / 2
    # Take TE x-position into account
    xpts *= xte

    # Powers to raise coefficients to
    pwrs = (1/2, 3/2, 5/2, 7/2, 9/2, 11/2)
    # Make [[1,1,1,1],[2,2,2,2],...] style array
    xptsgrid = np.meshgrid(np.arange(len(pwrs)), xpts)[1]
    # Evaluate points with concise matrix calculations. One x-coordinate is
    # evaluated for every row in xptsgrid
    evalpts = lambda cf: np.sum(cf*xptsgrid**pwrs, axis=1)
    # Move into proper order: start at TE, over bottom, then top
    # Avoid leading edge pt (0,0) being included twice by slicing [1:]
    ycoords = np.append(evalpts(cf_pre)[::-1], evalpts(cf_suc)[1:])
    xcoords = np.append(xpts[::-1], xpts[1:])

    # Return 2D list of coordinates [[x,y],[x,y],...] by transposing .T
    # Return lower surface then upper surface
    return np.array((xcoords, ycoords)).T


def params_to_coord(params):
    '''wrapper function to convert parsec parameters
        to coordinates and get only y coordinates'''
    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    yle = 0.0
    xte = 1.0
    yte = 0.0

    rle = params[0]

    # Pressure (lower) surface parameters 
    x_pre = params[1]
    y_pre = params[2]
    d2ydx2_pre = params[3]
    th_pre = params[4]

    # Suction (upper) surface parameters
    x_suc = params[5]
    y_suc = params[6]
    d2ydx2_suc = params[7]
    th_suc = params[8]
    cf_pre = pcoef(xte,yte,rle,x_pre,y_pre,d2ydx2_pre,th_pre,'pre')
    cf_suc = pcoef(xte,yte,rle,x_suc,y_suc,d2ydx2_suc,th_suc,'suc')
    coord = ppoints(cf_pre, cf_suc, xte=1.0)
    y_coord = coord[:,1]

    return y_coord

# Added Code
def read_data_csv():
    '''read data and compute y coordinate from the given csv file'''
    # extract data from csv file
    path_csv = Path(__file__).parent.joinpath('INPUT.csv')
    data=pd.read_csv(path_csv, header=None)

    # get x coordinate
    x_coord_data = data.iloc[0].to_numpy()

    # remove 'x coordinate', and NaN
    x_coord_data = np.delete(x_coord_data,[0])
    x_coord_data = np.delete(x_coord_data,[-1,-2,-3,-4])

    # replace '1' with 1 on the first element
    x_coord_data[0] = 1.0
    x_coord_data = x_coord_data.astype(np.float64)

    # index to indicate separation of different data
    ind = np.where(x_coord_data == 0)
    ind = int(ind[0])

    # create coordinate for upper and lower surface
    x_upper_coord = x_coord_data[:ind+1]
    x_lower_coord = x_coord_data[:ind+1]

    # get y coordinate
    y_coord_data = data.iloc[1:].to_numpy()

    # get airfoil label
    airfoil_label = y_coord_data[:,0]

    # remove the cl,cd,volume coefficient and fist line which show type of y coordinate
    y_coord_data = np.delete(y_coord_data,[0,-1,-2,-3,-4],axis=1)
    y_coord_data = np.delete(y_coord_data,[0],axis=0)

    # separate data - mean chord and thickness
    y_mean_cord_coord = y_coord_data[:,:ind+1]
    y_mean_cord_coord = y_mean_cord_coord.astype(np.float64)
    y_thick_coord = y_coord_data[:,ind+1:]
    y_thick_coord = y_thick_coord.astype(np.float64)

    # move the first column to the last column of the data for thickness
    # there seems to be error in the data set
    col_zero = y_thick_coord[:,0]
    col_zero = col_zero[:,np.newaxis]
    #y_thick_coord = np.delete(y_thick_coord,[0],axis=1)
    #y_thick_coord = np.hstack((y_thick_coord,col_zero))
    y_thick_coord = y_thick_coord/2

    # thickness data is labeled in reverse order from mean chord data
    # needs to be flipped
    y_thick_coord = np.fliplr(y_thick_coord) 

    # additional row is required for x=0
    y_thick_coord = np.hstack((y_thick_coord,col_zero))

    # create upper and lower mean chord data
    y_upper_coord = np.copy(y_mean_cord_coord)
    y_upper_coord = y_upper_coord + y_thick_coord
    y_lower_coord = np.copy(y_mean_cord_coord)
    y_lower_coord = y_lower_coord - y_thick_coord

    # combine the data such that it will match the output from p points
    # lower surface first then upper surface
    y_upper_coord = np.fliplr(y_upper_coord)
    x_upper_coord = np.flip(x_upper_coord)
    y_coord_data = np.hstack((y_lower_coord,y_upper_coord[:,1:]))
    x_coord_data = np.concatenate((x_lower_coord,x_upper_coord[1:]))

    return y_coord_data,x_coord_data,airfoil_label


def penalty(opt_params,opt_ind):
    '''compute penalty, which is sum of square root of difference between the data and
        coordinates generated from the parameters'''
    y_data = y_coord_data[opt_ind,:]
    y_params = params_to_coord(opt_params)
    
    pen_elem = np.square(y_data-y_params)
    pen = np.sum(pen_elem)

    return pen

# Main code
y_coord_data,x_coord_data,airfoil_label = read_data_csv()
airfoil_label = airfoil_label[1:,np.newaxis]
r,_ = y_coord_data.shape
params_0 = np.zeros(9) + 0.05

# preallocate parameter array
# last column is for the error
opt_params = np.zeros([r,10])

t1 = time()

for i in range(r):
    print(str(i+1)+'/'+str(r))
    bnds =((1e-6,np.inf),(1e-6,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(1e-6,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
    res = minimize(penalty, params_0, args=(i), tol=1e-6, method = 'SLSQP', options={'maxiter': 5000}, bounds=bnds)
    if res.success == False:
        print("Optimization Failed")
    opt_params[i,0:9] = np.array(res.x)
    opt_params[i,-1] = res.fun

t2 = time()
elapsed = t2 - t1
print(elapsed)

# plot the best case and the worst case
ind_best = np.argmin(opt_params[:,-1])
ind_worst = np.argmax(opt_params[:,-1])
y_params_best = opt_params[ind_best,:-1]
y_params_best = params_to_coord(y_params_best)
y_params_worst = opt_params[ind_worst,:-1]
y_params_worst = params_to_coord(y_params_worst)
y_data_best = y_coord_data[ind_best,:]
y_data_worst = y_coord_data[ind_worst,:]

fig,ax = plt.subplots(2)
ax[0].plot(x_coord_data,y_params_best, label='parameterized airfoil')
ax[0].plot(x_coord_data,y_data_best, label='actual airfoil')
ax[0].legend()
ax[0].set_title('Best Parameterization')
ax[0].set_aspect(5) # hard coded
ax[1].plot(x_coord_data,y_params_worst, label='parameterized airfoil')
ax[1].plot(x_coord_data,y_data_worst, label='actual airfoil')
ax[1].legend()
ax[1].set_title('Worst Parameterization')
ax[1].set_aspect(2) # hard coded
plt.tight_layout()

# save file
new_data = np.hstack((airfoil_label,opt_params))
path_output = Path(__file__).parent.joinpath('PARAMS_tmp.txt')
np.savetxt(path_output,new_data,fmt='%s')

plt.show()