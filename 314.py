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
    There is an important question that might be asked, which is that, if I'm
trying to show off (which I obviously am), why have I not bothered to reformulate
the fractional problem into a linear program? The answer is that the methods
are either not compatible with integer programming (i.e. Charnes-Cooper
transform), or requires more iterative steps than a third-party MIP solver would
allow (i.e. Dinkelbach’s Transform) and I really don't want to make my own
Branch and Bound algorithm when there are third party solvers availlable for free.
Also, both halves of the fraction are non-linear so would have to be linearised
anyway, so there really isn't any benefit of reformulating the fraction to a
linear program.
"""

import numpy as np
import scipy.optimize as op
import ctypes
import bokeh.plotting as plt
import matplotlib.pyplot as pyplot
#import pyomo.environ as pyo
#import cvxpy as cvx

class q_314():
    """The previous script put into a class. This removes the need for P being
    reshaped constantly, and makes all variables accessible without making
    everything a global variable"""
    def __init__(self, P_init=np.array([[250-75, 250]])):
        self.result = None
        
        if type(P_init) is not np.ndarray:
            self.P = self.make_P_init(P_init)
        else:
            self.P = P_init
        
        self.lengths = np.array([])
        self.perimeter = None
        self.area = None
        self.ratio = None
        
        
        #self.run()
    
    def run(self, buff_len = 3):
        """calls optimise and next_P until an optimum is reached"""
        self.optimise()
        self.shuffle()
        last_ratio = self.ratio
        while self.ratio >= last_ratio:
            print("length of P: {};\t ratio: {}".format(len(self.P), np.round(self.ratio, 8)))
            last_ratio = self.ratio
            self.next_P_lengths()
            self.optimise()
            self.shuffle()
        print("length of P: {};\t ratio: {}".format(len(self.P), np.round(self.ratio, 8)))
        print("optimum ratio is {}".format(np.round(last_ratio, 8)))
    
    def run2(self, end=200):
        self.optimise()
        res = [self.ratio]
        max_ratio = self.ratio
        max_len = self.P
        while len(self.P) <= end:
            self.next_P_lengths()
            self.optimise()
            res.append(self.ratio)
            if self.ratio > max_ratio:
                max_ratio = self.ratio
                max_len = len(self.P)
        print("optimum ratio is {}, with a P length of {}".format(np.round(max_ratio, 8), max_len))
        return res
    
    def make_P_init(self, n):
        """makes an initial set of P with n values"""
        x = np.linspace(0, 250, n+2)[1:-1]
        y = x*0 + 250 - np.linspace(0, len(x)-1, len(x))
        self.P = np.round(np.vstack((x, y)).T, 0)
        return self.P
    
    def flat_to_fat(self, fP):
        """converts the flattened P to its proper form. removes all np.insert
        this takes 1/3 of the time compared to np.insert"""
        len_P = len(fP) + 1
        fat_P = np.zeros((int(len_P/2), 2))
        fat_P[0] = np.array([fP[0], 250])
        fat_P[1:] = fP[1:].reshape(((int(len_P/2)-1), 2))
        return fat_P


    def find_ratio(self, P_in=None):
        """Returns the ratio of area to perimeter"""
        if P_in is not None:
            if len(np.shape(P_in)) == 1:
                self.P = self.flat_to_fat(P_in)
        P = self.P
        len_P = len(P)
        
        """find_lengths"""
        no_i = (P[-1, 1]-P[-1, 0])**2 > 0
        
        if no_i:
            L = np.zeros(len_P+1) # L = lengths of each line
            xe, ye = P[-1]
            L[-1] = np.sqrt((ye**2 + xe**2 - 2*ye*xe)*32)
        else:
            L = np.zeros(len_P)
        
        L[0] = np.abs(P[0][0]*8)
        if len_P != 1:
            dP = (P[1:] - P[:-1]).transpose()
            if no_i:
                L[1:-1] = np.sqrt(64*(dP[0]**2 + dP[1]**2)) # dx**2 + dy**2
            else:
                L[1:] = np.sqrt(64*(dP[0]**2 + dP[1]**2))
        
        self.lengths = L    # used in next_P_lengths and del_lengths
        perimeter = np.sum(L)
        self.perimeter = perimeter  # used in find_ratios and del_ratio
        
        """find area"""
        P0 = np.array([0, 250])
        
        # this vstack replacement cut time in shuffle by 20% for n=9
        P_e = np.zeros((len_P*2 + 2, 2))    #extended P (mirrors P about x=y to find area under the quarter)
        P_e[0] = P0
        P_e[-1] = P0[::-1]
        P_e[1:len_P+1] = P
        P_e[len_P+1:-1] = P.T[::-1].T[::-1]
        
        xe = P_e[:, 0]
        ye = P_e[:, 1]
        area = 2*np.sum((xe[1:]-xe[:-1])*(ye[1:]+ye[:-1]))
        self.area = area
        
        """find ratio"""
        self.ratio = area/perimeter
    
    def optimise(self):
        """optimises using scipy because fuck it.
        note that this function doesn't force x<=y. this is because it isn't
        necessary to make this bound explicit: it just does it itself. because
        x<=y isn't an explicit bound, the set of coordinates is an N dimensional
        square with side of {0, 250}, and N = 2*number of coordinates - 1.
        I don't know much about symmetry about hyperplanes, but I do know that
        N dimensional squares are always symmetric
        edit: even though the bound set is symmetric, the answer set apparently
        isn't. I don't understand why, but it isn't enough to simply round"""
        P = self.P
        flat_P = P.flatten()
        x0 = np.zeros(len(flat_P)-1)
        x0[0] = flat_P[0]
        x0[1:] = flat_P[2:]
        
        len_x = len(x0)
        b = [(0, 250)]
        b += [(0, 250), (0, 249)] * int(((len_x - 1)/2))
        
        
        self.result = op.minimize(self.func, x0, bounds=b, jac=True,
                                  options={"maxfun": 10**5})
        xres = np.round(self.result["x"])
        
        if len_x >= 7:
            while 1:
                ys = xres[1:].reshape((int((len_x-1)/2), 2))[:, 1]
                reps = ys[2:]<ys[:-2]
                if bool(reps.all()) is True:
                    break
                
                for n, i in enumerate(reps):
                    if bool(i) is False:
                        pos = (i+2)*2 +1
                        b[pos:] = [(0, 250), (0, xres[pos-1]-1)] * int((len_x-pos)/2)
                self.result = op.minimize(self.func, x0, bounds=b, jac=True,
                                  options={"maxfun": 10**5})
                xres = np.round(self.result["x"])
                    
        self.P = self.flat_to_fat(xres)
        self.find_ratio()
        

    def func(self, x):
        """scipy-friendly cost function for optimise()
        the jacobian must be the second returned variable, or scipy throws
        a hissy fit (if the jacobian is passed straight to .optimise(), it
        doesn't get recognised as a callable function by l-bfgs-b and it will
        assume the gradient is tacked on to this function"""
        self.P = self.flat_to_fat(x)
        self.find_ratio()
        return -self.ratio, -self.del_ratio()
    
    
    def next_P_lengths(self):
        """makes a new P by finding the longest length then splitting it.
        I think the new P will be further from optimum than the new P made by
        splitting the sharpest corner, but it's more straightforward to debug"""
        P = self.P
        i = 1+ np.argmax(self.lengths[1:]) # ignore the first length because that just adds stuff to y=250
        if i == 0:
            new_C = P[0]/2
        else:
            new_C = (P[i-1]+P[i])/2
        self.P = np.vstack((P[:i], new_C, P[i:]))
    
    def del_lengths(self):
        """finds the gradient of the lengths"""
        lengths = self.lengths/8 +0.001 #stops divide by zero errors
        
        P = self.P 
        no_i = (P[-1, 1]-P[-1, 0])**2 > 0  # test for if there is a coordinate on x=y
        
        if no_i:
            Pi = np.sum(P[-1])/2
            fP = np.vstack((np.array([0, 250]), P, np.array([1, 1])*Pi))
        else:
            fP = np.vstack((np.array([0, 250]), P))
        ddP = fP[1:] - fP[:-1]
        ddP2 = ddP/np.vstack((lengths, lengths)).T
        dels = (ddP2[:-1] - ddP2[1:]).flatten()
        
        del_out = np.zeros(len(dels)-1)
        del_out[0] = dels[0]
        del_out[1:] = dels[2:]
        if no_i:
            return del_out*8
        else:
            return np.append(del_out, (ddP[-1])/lengths[-1])*8
    
    def del_area(self):
        """Finds the gradient of the area
        Todo: allows for positive dx, negative dy, and dx>dy, for coordinates
        on the wrong side of y=x. Might be a flaw in the math?"""
        P = np.vstack((np.array([0, 250]), self.P))
        
        Pi = np.sum(P[-1])/2
        fP = np.vstack((P, np.array([1, 1])*Pi))
        
        dels_end = 4*np.array([P[-2, 1]-P[-1, 0], P[-1, 1]-P[-2, 0]])
        
        ddP = fP[2:] - fP[:-2]
        dels = 4*np.vstack((-ddP[:, 1], ddP[:, 0])).T
        flat_dels = dels.flatten()
        
        if len(P.flatten()) == 2:
            return dels_end[0]
        else:
            del_out = np.zeros(len(flat_dels)-1)
            del_out[0] = flat_dels[0]
            del_out[1:] = flat_dels[2:]
            del_out[-2:] = dels_end
            return del_out
        
    def del_ratio(self, P_in=None):
        """finds the jacobian of the ratio. can be used for scipy optimisation
        or for linearisation"""
        if P_in is not None:
            self.P = self.flat_to_fat(P_in)
        
        self.find_ratio()
        length = self.perimeter
        area = self.area
        dA = self.del_area()
        dL = self.del_lengths()
        del_r = (dA * length - area * dL)/(length**2)
        if type(del_r) is np.ndarray:
            return del_r
        else:
            return np.array([del_r])
    
        
    def shuffle(self):
        """shuffles the coordinates up and down by 1 until and optimum is
        found. This might be needed because I'm not 100% sure that the set
        is symmetric, but I am sure that it's a cross-shaped set because the
        problem is a 2d shadow inside the values set, so it's pretty easy
        to just how the potential values connect together. If I'm wrong,
        I'll make this function, but only if I'm wrong.
        edit: it appears that I was wrong. simple rounding caused the ratio
        to decrease when it should have been increasing. this function fixes that,
        but now the script is very slow
        the computational time for this function is exponential. there are two
        options that should be more efficient: when altering a point, also alter
        the neighbouring points; branch and bound; or run through the list altering
        points sequentially. all of these should be faster than exponential."""
        
        # collect the unrounded P from the optimise results
        iP = self.result['x'].astype(int)   # rounds P down to nearest integer
        len_iP = len(iP)
        
        # this is pretty gross, but the results are currently flattened by 
        # order=C, and the new ratio needs P flattened by order=F
        P = self.flat_to_fat(iP)
        len_P = len(P)
        iP = iP*0
        FfP = P.flatten(order="F")
        iP[:len_P] = FfP[:len_P]
        iP[len_P:] = FfP[len_P+1:]
        
        
        ratios = np.zeros(1 + 2**len_iP)   # results storage
        
        sel_array = np.zeros(len(iP))
        def s_P(i):
            """local worker function. turns i into a binary number, seperates
            each digit, packages the number into a numpy array with same shape
            as flattened P, and adds the number to P"""
            bin_i = bin(i)[2:]
            bin_array = np.array(list(bin_i), dtype=int)
            len_b = len(bin_i)
            sel_array[-len_b:] = bin_array
            s_f_P = iP + sel_array
            return s_f_P
        
        
        find_ratio_flat = self.find_ratio_flat
        
        ratios[0] = self.ratio
        
        for i in range(1, 2**len_iP):
            s_f_P = s_P(i)
            if np.any(s_f_P>250):
                ratios[i] = 0
            else:
                ratios[i] = find_ratio_flat(s_f_P, len_P)
        
        arg = np.argmax(ratios)
        self.ratio = ratios[arg]
        s_f_P = s_P(arg)
        self.P[:, 0] = s_f_P[:len_P]
        self.P[1:, 1] = s_f_P[len_P:]

    
    def find_ratio_flat(self, fP, len_P=None):
        """Returns the ratio of area to perimeter. accepts flattened P, but
        P must be flattened in fortran order instead of C order.
        Note: this function doesn't use self, and doesn't add to or read from
        the class dictionary. This is to save processing time, and because this
        function is a step towards building a ratio finder in C"""
        
        if len_P is None:
            len_P = int(len(fP)+1)/2
        
        pX = fP[:len_P]
        pY = np.zeros(len_P)
        pY[0] = 250
        pY[1:] = fP[len_P:]
        
        """find_lengths"""
        no_i = (pY[-1]-pX[-1])**2 > 0   #test for if there is a point on x=y
        
        if no_i:
            L = np.zeros(len_P+1) # L = lengths of each line
            L[-1] = np.sqrt((pY[-1]**2 + pX[-1]**2 - 2*pY[-1]*pX[-1])*32)
        else:
            L = np.zeros(len_P)
        
        L[0] = np.abs(pX[0]*8)
        if len_P != 1:
            dPX = pX[1:] - pX[:-1]
            dPY = pY[1:] - pY[:-1]
            if no_i:
                L[1:-1] = np.sqrt(64*(dPX**2 + dPY**2))
            else:
                L[1:] = np.sqrt(64*(dPX**2 + dPY**2))
        
        perimeter = np.sum(L)
        
        """find area"""
        xe = np.zeros((len_P*2 + 2))
        ye = np.zeros((len_P*2 + 2))
        #xe[0] = 0
        ye[0] = 250
        xe[1:len_P+1] = pX
        ye[1:len_P+1] = pY
        xe[len_P+1:-1] = pY[::-1]
        ye[len_P+1:-1] = pX[::-1]
        xe[-1] = 250
        #ye[-1] = 0
        
        area = 2*np.sum((xe[1:]-xe[:-1])*(ye[1:]+ye[:-1]))
        
        """find ratio"""
        return area/perimeter
            
            
                
            
        



def test_dels(n=None, eps=10**-7):
    """a function that brute-forces the gradients to test the math."""
    proj = q_314()
    if n is None:
        proj.P = np.array([[ 50., 250.], [100., 230.], [150., 215.], [200., 200.]])
    elif type(n)==int:
        proj.make_P_init(n)
    elif type(n) == np.ndarray:
        proj.P = n
    
    len_P = len(proj.P)*2 -1
    P_flat = np.zeros(len_P)
    P_flat[0] = proj.P[0, 0]
    P_flat[1:] = (proj.P).flatten()[2:]
    #P_flat = proj.P.flatten()
    
    d_lengths = np.zeros(len_P)
    d_areas = np.zeros(len_P)
    d_ratios = np.zeros(len_P)
    
    proj.find_ratio()
    del_l = np.round(proj.del_lengths(), 8)
    del_a = np.round(proj.del_area(), 8)
    del_r = np.round(proj.del_ratio(), 8)
    
    #print("lengths: {}".format(proj.lengths))
    
    for i in range(len_P):
        P_flat[i] += eps
        proj.P = (np.insert(P_flat, 1, 250)).reshape(int((len(P_flat)+1)/2),2)
        #proj.P = P_flat.reshape((4, 2))
        proj.find_ratio()
        d_lengths[i] += np.sum(proj.lengths)
        d_areas[i] += proj.area
        d_ratios[i] += proj.ratio
        
        P_flat[i] -= 2*eps
        proj.P = (np.insert(P_flat, 1, 250)).reshape(int((len(P_flat)+1)/2),2)
        #proj.P = P_flat.reshape((4, 2))
        proj.find_ratio()
        d_lengths[i] -= np.sum(proj.lengths)
        d_areas[i] -= proj.area
        d_ratios[i] -= proj.ratio
        
        P_flat[i] += eps
    
    d_l_brute = np.round(d_lengths/(2*eps), 8)
    d_a_brute = np.round(d_areas/(2*eps), 8)
    d_r_brute = np.round(d_ratios/(2*eps), 8)
    
    print("Lengths:")
    print("brute force: \t arithmatic: ")
    print(np.vstack((d_l_brute, del_l, np.abs(d_l_brute - del_l))).T)
    print("\n")
    print("Areas:")
    print("brute force: \t arithmatic: ")
    print(np.vstack((d_a_brute, del_a, np.abs(d_a_brute - del_a))).T)
    print("\n")
    print("Ratios:")
    print("brute force: \t arithmatic: ")
    print(np.vstack((d_r_brute, del_r, np.abs(d_r_brute - del_r))).T)

def plot(P):
    """Plots the quater graph made by P.
    If P is in the flattened form, it's reshaped to coordinate form"""
    if len(P.shape) == 1:
        P = (np.insert(P, 1, 250)).reshape(int((len(P)+1)/2),2)
    x, y = P.T
    graph = plt.figure()
    graph.line(np.hstack((x, y[::-1])), np.hstack((y, x[::-1])))
    plt.show(graph)


que = q_314(9)
que.optimise()
que.shuffle()
