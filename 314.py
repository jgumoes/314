# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:16:26 2018

@author: Jack

project euler problem 314
"""

import numpy as np
import scipy.optimize as op
import bokeh.plotting as plt


def make_P_init(n):
    """makes an initial set of P with n values"""
    x = np.linspace(0, 250, n+2)[1:-1]
    y = x*0 + 250
    return np.vstack((x, y)).T

def length(P):
    """outputs the perimeter, given coordinates P
    P0 is (x0, 250), and Pe is the last coordinate before the line of symmetry"""
    L = np.zeros(len(P)+1) # L = lengths of each line
    L[0] = P[0][0]*8
    xe, ye = P[-1]
    L[-1] = np.sqrt((ye**2 + xe**2 - 2*ye*xe)*32)
    
    if len(P) == 1:
        return np.sum(L)
    
    dP = (P[1:] - P[:-1]).transpose()
    L[1:-1] = np.sqrt(64*(dP[0]**2 + dP[1]**2))

    return np.sum(L)


def area(P):
    """Finds the total area using the trapezium rule"""
    A = np.zeros(len(P) + 1) # areas of each triangle segment

    yi = np.sum(P[-1])/2    # coordinates of the symmetry intercept
    P = np.vstack((P, np.array([yi, yi])))
    A[0] = P[0][0]*250*8
    
    xs = P.T[0]
    ys = P.T[1]
    
    A[1:] = (ys[1:]+ys[:-1])*(xs[1:]-xs[:-1])*4
    return np.sum(A) - 4*yi**2


def ratio(P):
    """Returns the ratio of area to perimeter"""
    L = length(P)
    A = area(P)
    return A/L

def opt(P):
    """optimises using scipy because fuck it"""
    P = P.flatten()
    x0 = np.zeros(len(P)-1)
    x0[0] = P[0]
    x0[1:] = P[2:]
    b = [(0, 250)]*len(x0)
    return op.minimize(func, x0, bounds=b)

def func(x):
    """scipy-friendly cost function"""
    P = (np.insert(x, 1, 250)).reshape(int((len(x)+1)/2),2)
    return -ratio(P)

# =============================================================================
# def next_P(P):
#     """finds the biggest corner, then splits it and optimizes again"""
# =============================================================================
    
    
def iterate_opt(P):
    """iterates opt() using the new coordinates found in next_P()"""    
    r = ratio(P)
    print(r)
    out_r = 0
    P_next = P
    while ((out_r < r) and (out_r != r)):
        r = out_r
        out = opt(make_P_init(len(P_next)+1))
        out_p = (out.x).round() 
        P_next = (np.insert(out_p, 1, 250)).reshape(int((len(out_p)+1)/2),2)
        out_r = ratio(P_next)
        print(len(P_next), out_r)


def plot(P):
    """Plots the quater graph made by P.
    If P is in the flattened form, it's reshaped to coordinate form"""
    if len(P.shape) == 1:
        P = (np.insert(P, 1, 250)).reshape(int((len(P)+1)/2),2)
    x, y = P.T
    graph = plt.figure()
    graph.line(np.hstack((x, y[::-1])), np.hstack((y, x[::-1])))
    plt.show(graph)
    
    

"""Junk ahoy!"""
# =============================================================================
# def update(P, F, F_l, X_L):
#     """Updates the value of P using gradient descent"""
#     # reshape P into a vector array of only the variables
#     f_P = P.flatten()
#     X = np.zeros(len(f_P)-1)
#     X[0] = f_P[0]
#     X[1:] = f_P[2:]
#     
#     X_f = X_L.flatten()
#     X_l = np.zeros(len(X_f)-1)
#     X_l[0] = X_f[0]
#     X_l[1:] = X_f[2:]
#     
#     #F = -ratio(P)  # the current function value
#     
#     fd = f_d(X) - F
#     bd = F - b_d(X)
#     
#     # finding the Jacobian matrix i.e. grad(-ratio)
#     # it looks messy as hell, but it's faster than using for loops
#     ineq = (fd>=0)*(bd<=0)    # element=1 if point is a minima
#     float_J = ((fd+bd)/2)*(1-ineq)  # stops non-zero derivatives appearing at minima
#     y = np.sum((X - X_l)*(F - F_l))/np.sum((F - F_l)**2) #  Barzilai-Borwein
#     J = (float_J*y).astype(int) # or maybe .round() ?
#     print(J)
#     float_J *= y
#     print(float_J)
#     J += (float_J!=0)*(J==0)*1*np.sign(float_J) # i can't remember why i did this
#     #print(J, ineq, fd, bd)
#     
#     X = X - J
#     lim = X > 250   # =1 if element is over 250 (out of bounds)
#     X = X*(1-lim) + 250*lim
#     #print(X)
#     P = (np.insert(X, 1, 250)).reshape(int((len(X)+1)/2),2)
#     
#     return P, np.max(J**2)
# 
# def update0(P, F):
#     """a function for the first iteration. is needed because the gamma varibale
#     is found using the gradient of the last iteration, which obviously doesn't
#     exist for the first one"""
#     # reshape P into a vector array of only the variables
#     f_P = P.flatten()
#     X = np.zeros(len(f_P)-1)
#     X[0] = f_P[0]
#     X[1:] = f_P[2:]
#     
#     #F = -ratio(P)  # the current function value
#     
#     fd = f_d(X) - F
#     bd = F - b_d(X)
#     
#     # finding the Jacobian matrix i.e. grad(-ratio)
#     # it looks messy as hell, but it's faster than using for loops
#     ineq = (fd>=0)*(bd<=0)    # element=1 if point is a minima
#     float_J = ((fd+bd)/2)*(1-ineq)  # stops non-zero derivatives appearing at minima
#     
#     J = np.sign(float_J)
#     #print(J, ineq, fd, bd)
#     
#     X = X - J
#     P = (np.insert(X, 1, 250)).reshape(int((len(X)+1)/2),2)
#     
#     return P, np.max(J**2)
#     
# def f_d(X):
#     """calculates the forward derivatives given X"""
#     f_X = np.eye(len(X)) + X
#     f_F = np.zeros(len(X))
#     for v in range(0, len(f_X)):
#         i = f_X[v]
#         P = (np.insert(i, 1, 250)).reshape(int((len(i)+1)/2),2)
#         f_F[v] = -ratio(P)
#     return f_F
# 
# def b_d(X):
#     """calculates the forward derivatives given X"""
#     f_X = X - np.eye(len(X))
#     f_F = np.zeros(len(X))
#     for v in range(0, len(f_X)):
#         i = f_X[v]
#         P = (np.insert(i, 1, 250)).reshape(int((len(i)+1)/2),2)
#         f_F[v] = -ratio(P)
#     return f_F
# 
# def opt(P):
#     """optimises P to maximise ratio, using update()"""
#     Ra = ratio(P)
#     #cost = 10
#     P, cost = update0(P, -Ra)
#     print(Ra, P, cost)
#     while cost > 0: # there's a chance this might osciallate around the answer
#         P, cost = update(P, -Ra, cost, P)
#         Ra = ratio(P)
#         print(Ra, P, cost)
# =============================================================================
