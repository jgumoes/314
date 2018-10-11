# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:16:26 2018

@author: Jack

project euler problem 314
"""

import numpy as np

def make_P_init(n):
    """makes an initial set of P with n values"""
    x = np.linspace(0, 250, n+2)[1:-1]
    y = x*0 + 250
    return np.vstack((x, y)).T

def lengths(P):
    """outputs the lengths of each line in the segment, given coordinates P
    P0 is (x0, 250), and Pe is the last coordinate before the line of symmetry"""
    L = np.zeros(len(P)+1) # L = lengths of each line
    L[0] = P[0][0]
    xe, ye = P[-1]
    L[-1] = np.sqrt((ye**2 + xe**2 - 2*ye*xe)/2)
    
    if len(P) == 1:
        return L
    
    dP = (P[1:] - P[:-1]).transpose()
    L[1:-1] = np.sqrt(dP[0]**2 + dP[1]**2)
    return L


def area(P, L):
    """the area under the segment, given coordinate P
    Finds the area using trigonometry
    This is 1/8 of the true area"""
    A = np.zeros(len(P) + 1) # areas of each triangle segment
    #PT = P.transpose()
    
    A[0] = P[0][0]*125
    A[-1] = L[-1]*np.sqrt(2)*(P[-1][0]+P[-1][1])/4
    
    P = P.flatten()
    
    if len(P) == 1:
        return A
    
    R = np.sqrt(np.sum(P**2, 1)) # distance of each point from the center
    S = (R[1:] + R[:-1] + L[1:-1])/2
    
    A[1:-1] = np.sqrt(S*(S-R[1:])*(S-R[:1])*(S-L[1:-1]))
    return A


def ratio(P):
    """Returns the ratio of area to perimeter"""
    L = lengths(P)
    A = area(P, L)
    return np.sum(A)/np.sum(L)

def update(P, F, F_l, X_l):
    """Updates the value of P using gradient descent"""
    # reshape P into a vector array of only the variables
    f_P = P.flatten()
    X = np.zeros(len(f_P)-1)
    X[0] = f_P[0]
    X[1:] = f_P[2:]
    
    #F = -ratio(P)  # the current function value
    
    fd = f_d(X) - F
    bd = F - b_d(X)
    
    # finding the Jacobian matrix i.e. grad(-ratio)
    # it looks messy as hell, but it's faster than using for loops
    ineq = (fd>=0)*(bd<=0)    # element=1 if point is a minima
    float_J = ((fd+bd)/2)*(1-ineq)  # stops non-zero derivatives appearing at minima
    y = np.sum((X - X_l)*(F - F_l))/np.sum((F - F_l)**2) #  Barzilai-Borwein
    J = np.int(float_J*y)
    J += (float_J!=0)*(J==0)*1*np.sign(float_J)
    #print(J, ineq, fd, bd)
    
    X = X - J
    lim = X > 250   # =1 if element is over 250 (out of bounds)
    X = X*(1-lim) + 250*lim
    #print(X)
    P = (np.insert(X, 1, 250)).reshape((int(len(X)+1/2), 2))
    
    return P, np.max(J**2)

def update0(P, F):
    """a function for the first iteration. is needed because the gamma varibale
    is found using the gradient of the last iteration, which obviously doesn't
    exist for the first one"""
    # reshape P into a vector array of only the variables
    f_P = P.flatten()
    X = np.zeros(len(f_P)-1)
    X[0] = f_P[0]
    X[1:] = f_P[2:]
    
    #F = -ratio(P)  # the current function value
    
    fd = f_d(X) - F
    bd = F - b_d(X)
    
    # finding the Jacobian matrix i.e. grad(-ratio)
    # it looks messy as hell, but it's faster than using for loops
    ineq = (fd>=0)*(bd<=0)    # element=1 if point is a minima
    float_J = ((fd+bd)/2)*(1-ineq)  # stops non-zero derivatives appearing at minima
    
    J = np.sign(float_J)
    #print(J, ineq, fd, bd)
    
    X = X - J
    P = (np.insert(X, 1, 250)).reshape((int(len(X)+1/2), 2))
    
    return P, np.max(J**2)
    
def f_d(X):
    """calculates the forward derivatives given X"""
    f_X = np.eye(len(X)) + X
    f_F = np.zeros(len(X))
    for v in range(0, len(f_X)):
        i = f_X[v]
        P = (np.insert(i, 1, 250)).reshape((int((len(i)+1/2)/2), 2))
        f_F[v] = -ratio(P)
    return f_F

def b_d(X):
    """calculates the forward derivatives given X"""
    f_X = X - np.eye(len(X))
    f_F = np.zeros(len(X))
    for v in range(0, len(f_X)):
        i = f_X[v]
        P = (np.insert(i, 1, 250)).reshape((int((len(i)+1/2)/2), 2))
        f_F[v] = -ratio(P)
    return f_F

def opt(P):
    """optimises P to maximise ratio, using update()"""
    Ra = ratio(P)
    #cost = 10
    P, cost = update0(P, -Ra)
    while cost > 0: # there's a chance this might osciallate around the answer
        P, cost = update(P, -Ra)
        Ra = ratio(P)
        print(Ra, P, cost)
