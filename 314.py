# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:16:26 2018

@author: Jack

project euler problem 314

Note: this attempt is not the first: the first attempt tried to find the largest
corner and squash it down. The idea is that sharp corners add more perimeter
than area, so the ratio can be improved by squashing the to remove some of the
perimeter.
    The current attempt was inspired when I was reading into Tikhonov
regularization. The script optimizes an initial set of coordinates, then
generates a new set of coordinates by finding the largest corner and splitting
it. If this process continues indefinitely and without being bounded to integer
space, the end result would be a continuous curve.
    There are a couple of assumptions that grew out of the squashing-corner
idea, which is that any section of perimeter that crosses a line of symmetry
should be tangential to that line of symmetry. However, this appears to be
not strictly true: it appears to be true for diagonal (x=y) symmetry, but an
optimum y value for the vertical symmetry (x=0) is over 250, so y1=250 is still
valid thank christ.
    A better approach than splitting corners might be to put a corner into a
long flat, and let it grow into an optimally-shaped corner. However, given the
optimisation procedures, it shouldn't really matter how the new coordinates are
placed because if a new coordinate is badly placed, it'll be reshuffled anyway.
The coordinate placement routine is mostly just to try to take some load of the
optimisation routines.
"""

import numpy as np
import scipy.optimize as op
import bokeh.plotting as plt


class q_314():
    """The previous script put into a class. This removes the need for P being
    reshaped constantly, and makes all variables accessible without making
    everything a global variable"""
    def __init__(self, P_init=np.array([[250-75, 250]])):
        self.result = None
        self.old_result = None
        
        if type(P_init) is not np.ndarray:
            self.P = self.make_P_init(P_init)
        else:
            self.P = P_init
        self.P_old = self.P*0
        
        self.lengths = np.array([])
        self.perimeter = None
        self.area = None
        self.ratio = None
    
    def run(self):
        """one day, this will grow into a beautiful function, but not yet"""
        self.optimise()
        
    
    def find_lengths(self):
        """outputs the perimeter, given coordinates P
        P0 is (x0, 250), and Pe is the last coordinate before the line of symmetry
        TODO: if doesn't work, re-write so that yi is added to P like in find_area()"""
        P = self.P
        no_i = np.diff(P[-1])[0] > 0    # test for if there is a coordinate on x=y
        
        if no_i:
            L = np.zeros(len(P)+1) # L = lengths of each line
            xe, ye = P[-1]
            L[-1] = np.sqrt((ye**2 + xe**2 - 2*ye*xe)*32)
        else:
            L = np.zeros(len(P))
        
        L[0] = P[0][0]*8
        if len(P) != 1:
            dP = (P[1:] - P[:-1]).transpose()
            if no_i:
                L[1:-1] = np.sqrt(64*(dP[0]**2 + dP[1]**2))
            else:
                L[1:] = np.sqrt(64*(dP[0]**2 + dP[1]**2))
        
        self.lengths = L
        self.perimeter = np.sum(L)
    
    def find_area(self):
        """Finds the total area using the trapezium rule"""
        P = self.P
        no_i = np.diff(P[-1])[0] > 0    # test for if there is a coordinate on x=y
        
        A = np.zeros(len(P) + no_i) # areas of each triangle segment
        
        if no_i:
            yi = np.sum(P[-1])/2    # coordinates of the symmetry intercept
            P = np.vstack((P, np.array([yi, yi])))
        else:
            yi = P[-1][1]
        A[0] = P[0][0]*250*8
        
        xs = P.T[0]
        ys = P.T[1]
        
        A[1:] = (ys[1:]+ys[:-1])*(xs[1:]-xs[:-1])*4
        self.area = np.sum(A) - 4*yi**2

    def make_P_init(self, n):
        """makes an initial set of P with n values"""
        x = np.linspace(0, 250, n+2)[1:-1]
        y = x*0 + 250
        self.P = np.vstack((x, y)).T
        return self.P

    def find_ratio(self):
        """Returns the ratio of area to perimeter"""
        #if len(P.shape)==1:
        #    P = (np.insert(P, 1, 250)).reshape(int((len(P)+1)/2),2)
        self.find_area()
        self.find_lengths()
        self.ratio = self.area/self.perimeter
    
    def optimise(self):
        """optimises using scipy because fuck it"""
        #if type(P) is int:
        #    P = make_P_init(P)
        P = self.P
        flat_P = P.flatten()
        x0 = np.zeros(len(flat_P)-1)
        x0[0] = flat_P[0]
        x0[1:] = flat_P[2:]
        b = [(0, 250)]*len(x0)
        self.result = op.minimize(self.func, x0, bounds=b, jac=True,
                                  options={"maxfun": 10**5})
        xres = np.round(self.result["x"])
        if type(xres) is int or type(xres) is float:
            len_xres = 1
        else:
            len_xres = len(xres)
        self.P = (np.insert(xres, 1, 250)).reshape(int((len_xres+1)/2),2)
        

    def func(self, x):
        """scipy-friendly cost function for optimise()
        the jacobian must be the second returned variable, or scipy throws
        a hissy fit (if the jacobian is passed straight to .optimise(), it
        doesn't get recognised as a callable function by l-bfgs-b and it will
        assume the gradient is tacked on to this function"""
        self.P = (np.insert(x, 1, 250)).reshape(int((len(x)+1)/2),2)
        self.find_ratio()
        return -self.ratio, -self.del_ratio()
    
    def next_P(self):
        """Makes a new P by finding the sharpest corner and splitting it.
        If the end coordinate is too close to the line, it gets trimmed and ignored,
            and the function will make a new P with the same length as the
            old P. There is a risk that the routine could get stuck at one length,
            so there should be a check to see if the length of the coordinates
            isn't growing, then use a different method to make new coordinates
            TODO: store previous coordinate sizes to test for sticking"""
        P = self.P
        self.P_old = P          # good to have just in case
        
        yi = np.sum(P[-1])/2    # coordinates of the symmetry intercept
        
# =============================================================================
#         if np.diff(P[-1]) < 2:  # trim off the end coordinate if it's too close to x=y
#             #P_split = P[:-1]
#             """this could cause the script to stuck at one length"""
#             P_split = np.vstack((P[:-1], np.array([yi, yi])))
#         else:
#             P_split = np.vstack((P, np.array([yi, yi])))
# =============================================================================
        
        P_split = np.vstack((P, np.array([yi, yi])))
        
        Xs, Ys = np.vstack((np.array([0, 250]), P_split)).T
        grads = (Ys[1:]-Ys[:-1])/(Xs[1:]-Xs[:-1])   # gradients of each line section
        thetas = np.arctan(grads)                   # angle of each line section with respect to x axis
        angles = np.pi - thetas[:-1] + thetas[1:]   # the angle between each line segment
        
        pos_max = angles.argmin()   # position of the sharpest corner
                                    # the index should line up if using P_split
        #print(pos_max)
        #print(angles)
        new_X = np.zeros(len(P_split[1:-1]) + 1)    # i'm using P_split because if a coordinate is
        new_Y = np.zeros(len(P_split[1:-1]) + 1)    # too close to y=x, it won't integerise nicely
        
        
        if (pos_max <= (len(P_split)-3)):
            Xs = Xs[1:-1]
            Ys = Ys[1:-1]
            new_X[:pos_max] = Xs[:pos_max]
            new_X[pos_max:pos_max+2] = np.round((Xs[pos_max:pos_max+2] + Xs[pos_max-1:pos_max+1])/2)
            new_X[pos_max+2:] = Xs[pos_max+1:]
            
            new_Y[:pos_max] = Ys[:pos_max]
            new_Y[pos_max:pos_max+2] = np.round((Ys[pos_max:pos_max+2] + Ys[pos_max-1:pos_max+1])/2)
            new_Y[pos_max+2:] = Ys[pos_max+1:]
        else:
            Xs = Xs[1:]
            Ys = Ys[1:]
            new_X[:-2] = Xs[:-3]
            new_X[-2:] = np.round((Xs[-3:-1] + Xs[-4:-2])/2)
            #new_X[-1] = Xs[-1]
            
            new_Y[:-2] = Ys[:-3]
            new_Y[-2:] = np.round((Ys[-3:-1] + Ys[-4:-2])/2)
            #new_Y[-1] = Ys[-1]
            
        self.P = (np.vstack((new_X, new_Y))).T
    
    def next_P_lengths(self):
        """makes a new P by finding the longest length then splitting it.
        to be used if next_P() gets stuck in a loop
        TODO: make"""
    
    def del_lengths(self):
        """finds the gradient of the lengths"""
        lengths = self.lengths/8
        
        P = self.P
        no_i = np.diff(P[-1])[0] > 0    # test for if there is a coordinate on x=y
        
        if no_i:
            Pi = np.sum(P[-1])/2
            fP = np.vstack((np.array([0, 250]), P, np.array([1, 1])*Pi))
        else:
            fP = np.vstack((np.array([0, 250]), P))
        ddP = fP[1:] - fP[:-1]
        ddP2 = ddP/np.vstack((lengths, lengths)).T
        dels = (ddP2[:-1] - ddP2[1:]).flatten()
        #print(ddP)
        #print(8*(ddP)/lengths[-1])
        del_out = np.zeros(len(dels)-1)
        del_out[0] = dels[0]
        del_out[1:] = dels[2:]
        if no_i:
            return del_out*8
        else:
            return np.append(del_out, (ddP[-1])/lengths[-1])*8
    
    def del_area(self):
        """Finds the gradient of the area"""
        P = self.P
        no_i = np.diff(P[-1])[0] > 0    # test for if there is a coordinate on x=y
        
        Pi = np.sum(P[-1])/2
        fP = np.vstack((np.array([0, 250]), P, np.array([1, 1])*Pi))
        
        if no_i:
            dels_end = 4*np.array([fP[-3, 1]-fP[-2, 0], fP[-2, 1]-fP[-3, 0]])
        else:
            #fP = np.vstack((np.array([0, 250]), P))
            dels_end = 4*np.array([P[-2, 1], -P[-2, 0]])
        
        ddP = fP[2:] - fP[:-2]
        #print("ddP: {}".format(ddP*4))
        dels = 4*np.vstack((-ddP[:, 1], ddP[:, 0])).T
        #print("del_area: {}".format(4*np.vstack((-ddP[:, 1], ddP[:, 0]))))
        #print("flat del_area: {}".format((4*np.vstack((-ddP[:, 1], ddP[:, 0])).T).flatten()))
        
        #dels_end = 4*np.array([fP[-3, 1]-fP[-2, 0], fP[-2, 1]-fP[-3, 0]])
        flat_dels = dels.flatten()
        if len(P.flatten()) == 2:
            return dels_end[0]
        else:
            del_out = np.zeros(len(flat_dels)-1)
            del_out[0] = flat_dels[0]
            del_out[1:] = flat_dels[2:]
            del_out[-2:] = dels_end #del_out[-2:] - np.sum(P[-1])
            return del_out
        
    def del_ratio(self, P_in=None):
        """finds the jacobian of the ratio. can be used for scipy optimisation
        or for linearisation"""
        if P_in is not None:
            P = (np.insert(P_in, 1, 250)).reshape(int((len(P_in)+1)/2),2)
            self.P = P
        
        self.find_lengths()
        self.find_area()
        length = self.perimeter
        area = self.area
        dA = self.del_area()
        dL = self.del_lengths()
        del_r = (dA * length - area * dL)/(length**2)
        if type(del_r) is np.ndarray:
            return del_r
        else:
            return np.array([del_r])



def test_dels(n=None, eps=10**-7):
    """a function that brute-forces the gradients to test the math."""
    proj = q_314()
    if n is None:
        proj.P = np.array([[ 50., 250.], [100., 230.], [150., 215.], [200., 200.]])
    else:
        proj.make_P_init(n)
    
    len_P = len(proj.P)*2 -1
    P_flat = np.zeros(len_P)
    P_flat[0] = proj.P[0, 0]
    P_flat[1:] = (proj.P).flatten()[2:]
    #P_flat = proj.P.flatten()
    
    d_lengths = np.zeros(len_P)
    d_areas = np.zeros(len_P)
    d_ratios = np.zeros(len_P)
    
    proj.find_lengths()
    del_l = proj.del_lengths()
    del_a = proj.del_area()
    del_r = proj.del_ratio()
    
    #print("lengths: {}".format(proj.lengths))
    
    for i in range(len_P):
        P_flat[i] += eps
        proj.P = (np.insert(P_flat, 1, 250)).reshape(int((len(P_flat)+1)/2),2)
        #proj.P = P_flat.reshape((4, 2))
        proj.find_lengths()
        proj.find_lengths()
        proj.find_ratio()
        d_lengths[i] += np.sum(proj.lengths)
        d_areas[i] += proj.area
        d_ratios[i] += proj.ratio
        
        P_flat[i] -= 2*eps
        proj.P = (np.insert(P_flat, 1, 250)).reshape(int((len(P_flat)+1)/2),2)
        #proj.P = P_flat.reshape((4, 2))
        proj.find_lengths()
        proj.find_lengths()
        proj.find_ratio()
        d_lengths[i] -= np.sum(proj.lengths)
        d_areas[i] -= proj.area
        d_ratios[i] -= proj.ratio
        
        P_flat[i] += eps
    
    print("Lengths:")
    print("brute force: {} \n arithmatic: {}".format(d_lengths/(2*eps), del_l))
    print("\n")
    print("Areas:")
    print("brute force: {} \n arithmatic: {}".format(d_areas/(2*eps), del_a))
    print("\n")
    print("Ratios:")
    print("brute force: {} \n arithmatic: {}".format(d_ratios/(2*eps), del_r))
    print("\n")


def plot(P):
    """Plots the quater graph made by P.
    If P is in the flattened form, it's reshaped to coordinate form"""
    if len(P.shape) == 1:
        P = (np.insert(P, 1, 250)).reshape(int((len(P)+1)/2),2)
    x, y = P.T
    graph = plt.figure()
    graph.line(np.hstack((x, y[::-1])), np.hstack((y, x[::-1])))
    plt.show(graph)
    
def opt_i():
    """a function to determine if a point on x=y can be more optimum than a
    tangent across the intercept.
    xn, yn are the last coordinates assuming there is no corner on the intercept.
    i is a corner point on the intercept
    vn are variables made without i, vi are variables made with i
    v_dx are variables made replacing yn with xn+dx"""
    import sympy as sy
    sy.init_printing()
    xn, yn, i, dx, S = sy.symbols("xn yn i dx S", positive=True)
    Li = sy.sqrt((i-xn)**2 + (i-yn)**2)
    Ai = (yn-xn)*(sy.sqrt(2) + 2*i - (xn+yn))/2*sy.sqrt(2)
    Li_dx = sy.sqrt((i-xn)**2 + (i-xn-dx)**2)
    Ai_dx = (dx)*(sy.sqrt(2) + 2*i - (2*xn+dx))/2*sy.sqrt(2)
    Ri = Ai/Li
    Ri_dx = Ai_dx/Li_dx
    
    Rn = (yn-xn)/2*sy.sqrt(2)
    Rn_dx = (dx)/2*sy.sqrt(2)
    
    solve = sy.solve(sy.diff(Ri, i), i)
    solve_dx = sy.solve(sy.diff(Ri_dx, i), i)
    