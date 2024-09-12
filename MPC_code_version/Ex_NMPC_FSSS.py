# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:11:54 2016

@author: marcovaccari
"""
from __future__ import division
from past.utils import old_div
from casadi import *
from casadi.tools import *
from matplotlib import pylab as plt
import math
import scipy.linalg as scla
import numpy as np
from Utilities import*
from Model import *

### 1) Simulation Fundamentals

# 1.1) Simulation discretization parameters
Nsim = 100 # Simulation length

N = 20    # Horizon

h = 5/60 # Time step

# 3.1.2) Symbolic variables
xp = SX.sym("xp", 4) # process state vector       
x = SX.sym("x", 4)  # model state vector          
u = SX.sym("u", 2)  # control vector              
y = SX.sym("y", 3)  # measured output vector      
d = SX.sym("d", 0)  # disturbance                     

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 2) Process and Model construction 

# 2.1) Process Parameters


# State map
def User_fxp_Cont(x,t,u,pxp,pxmp):    # States
    V, X, S, CO2 = x[0], x[1], x[2], x[3]

    # Manipulated variables
    Fin = u[0]
    Fout = u[1]

    # Parameters
    Sin = 200
    Q = 2
    V_total = 2.7  # L
    CO2in = 0.04  # 100*volume fraction of CO2 in feed

    Y_XS = 0.4204
    mu_max = 0.1945
    Ks = 0.0070
    Y_CO2X = 0.5430
    kd = 0.0060

    # Define the rate
    mu = mu_max * (S / (Ks + S))

    # Differential equations
    dV = Fin - Fout
    dX = -X * (Fin / V) + mu * X - kd * X  # Biomass
    dS = (Sin - S) * (Fin / V) - mu * X / Y_XS  # Substrate
    dCO2 = ((CO2in - CO2) * Q + mu * X / Y_CO2X) / (V_total - V)  # CO2

    # Output
    fx_p = ca.vertcat(dV, dX, dS, dCO2) 
    
    
    return fx_p

Mx = 10 # Number of elements in each time step 

# Output map
def User_fyp(x,u,t,pyp,pymp):
    fy_p = ca.vertcat\
    (\
    x[0],\
    x[1],\
    x[3] \
    )

    return fy_p

# White Noise
#R_wn = 1e-8*np.array([[1.0, 0.0], [0.0, 1.0]]) # Output white noise covariance matrix


# 2.2) Model Parameters
    
# State Map
def User_fxm_Cont(x,u,d,t,px):    
    # States
    V, X, S, CO2 = x[0], x[1], x[2], x[3]

    # Manipulated variables
    Fin = u[0]
    Fout = u[1]

    # Parameters
    Sin = 200
    Q = 2
    V_total = 2.7  # L
    CO2in = 0.04  # 100*volume fraction of CO2 in feed

    Y_XS = 0.4204
    mu_max = 0.1945
    Ks = 0.0070
    Y_CO2X = 0.5430
    kd = 0.0060

    # Define the rate
    mu = mu_max * (S / (Ks + S))

    # Differential equations
    dV = Fin - Fout
    dX = -X * (Fin / V) + mu * X - kd * X  # Biomass
    dS = (Sin - S) * (Fin / V) - mu * X / Y_XS  # Substrate
    dCO2 = ((CO2in - CO2) * Q + mu * X / Y_CO2X) / (V_total - V)  # CO2

    # Output
    fx_p = ca.vertcat(dV, dX, dS, dCO2) 
    
    return fx_p

# Output Map
def User_fym(x,u,d,t,py):
    fy_p = ca.vertcat\
    (\
    x[0],\
    x[1],\
    x[3] \
    )

    return fy_p
    
Mx = 10 # Number of elements in each time step 

# 2.3) Disturbance model for Offset-free control
offree = "no" # TODO: this is also to be changed if you wnat to activate the offsetfree potentiality

# 2.4) Initial condition
xs_CSTR = np.array([1.0, 10.0, 0.0, 0.0])
us_CSTR = np.array([0.0, 0.0])
x0_m = np.array([1.2, 3.0, 10.0, 0.1]) * xs_CSTR.ravel()
x0_p = np.array([1.2, 3.0, 10.0, 0.1]) * xs_CSTR.ravel()
u0 = np.array([0.0, 0.0])
#dhat0 = np.array([0, 0.1]) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 3) State Estimation
#### Extended Kalman filter tuning params ###################################
ekf = True # Set True if you want the Kalman filter
nx = x.size1()
ny = y.size1()
nd = d.size1()
Qx_kf = 1.0e-10*np.eye(nx)
Qd_kf = 1.0*np.eye(nd)
Q_kf = scla.block_diag(Qx_kf, Qd_kf)
R_kf = 1.0e-4*np.eye(ny)
P0 = 1.0*np.eye(nx+nd)  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### 4) Steady-state and dynamic optimizers

# 4.1) Setpoints
def defSP(t):
    """
    SUMMARY:
    It constructs the setpoints vectors for the steady-state optimisation 
    
    SYNTAX:
    assignment = defSP(t)
  
    ARGUMENTS:
    + t             - Variable that indicates the current time
    
    OUTPUTS:
    + ysp, usp, xsp - Input, output and state setpoint values      
    """ 
    xsp = np.array([1.0, 10.0, 0.0, 0.0]) # State setpoints  
    ysp = np.array([1.0, 10.0, 0.0]) # Output setpoint
    usp = np.array([0.0, 0.0]) # Control setpoints

    return [ysp, usp, xsp]
    
# 4.2) Bounds constraints
## Input bounds
umin = np.array([0.0, 0.0])
umax = np.array([0.4, 0.4])

## State bounds
xmin = np.array([0.5, 0.0, 0.0, 0.0])
xmax = np.array([2.0, 20, 5.0, 100.0])

## Output bounds
ymin = np.array([0.5, 0.0, 0.0])
ymax = np.array([2.0, 20, 5.0])

## Disturbance bounds
dmin = -100*np.ones((d.size1(),1))
dmax = 100*np.ones((d.size1(),1))

# 4.3) Steady-state optimization : objective function
Qss = np.zeros([3,3]) #Output matrix
Rss = np.zeros([2,2]) # Control matrix

# 4.4) Dynamic optimization : objective function 
Q = np.identity(4)# np.zeros([4,4])
Q[1,1] = 1.0
Q[2,2] = 10.0
R = np.array([[0.1, 0.0], [0.0, 0.1]])

slacksG = False
slacksH = False

# slacks = True

# Ws = np.eye(4)


pathfigure = 'MPC_Images/'

# def User_fobj_Coll(x,u,y,xs,us,ys,s_Coll):
#     """
#     SUMMARY:
#     It constructs the objective function for dynamic optimization

 

#     SYNTAX:
#     assignment = User_fobj_Coll(x,u,y,xs,us,ys,s_Coll)

 

#     ARGUMENTS:
#     + x,u,y         - State, input and output variables
#     + xs,us,ys      - State, input and output stationary variables
#     + s_Coll        - Internal state variables

 

#     OUTPUTS:
#     + obj         - Objective function
#     """    

#     Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#     R = np.array([[0.1, 0.0], [0.0, 0.1]])
        
#     obj = 1/N*(xQx(x,Q) + xQx(u,R))

#     return obj    

# Collocation = True