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
import bokeh.plotting as plt
import pyomo.environ as pyo
import cvxpy as cvx


class q_314():
    """The previous script put into a class. This removes the need for P being
    reshaped constantly, and makes all variables accessible without making
    everything a global variable"""
    def __init__(self, P_init=np.array([[250-75, 250]])):
        self.result = None
        self.old_result = None      # is this being used?
        
        if type(P_init) is not np.ndarray:
            self.P = self.make_P_init(P_init)
        else:
            self.P = P_init
        self.P_old = self.P*0
        
        self.lengths = np.array([])
        self.perimeter = None
        self.area = None
        self.areas = np.array([])
        self.ratio = None
        self.last_ratio = 0
        
        self.dR = None
        self.flat_P = None
        
        #self.run()
    
    def run(self):
        """calls optimise and next_P until an optimum is reached"""
        self.optimise()
        while self.ratio > self.last_ratio:
            print("length of P: {};\t ratio: {}".format(len(self.P), np.round(self.ratio, 8)))
            self.last_ratio = self.ratio
            self.next_P_lengths()
            self.optimise()
        print("length of P: {};\t ratio: {}".format(len(self.P), np.round(self.ratio, 8)))
        print("optimum ratio is {}".format(np.round(self.last_ratio, 8)))
        #self.MIP()
        
    
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
        """Finds the total area using the trapezium rule
        Todo: this function allows positive areas when perimeter double-backs
        on itself"""
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
        self.areas = A
        self.area = np.sum(A) - 4*yi**2

    def make_P_init(self, n):
        """makes an initial set of P with n values"""
        x = np.linspace(0, 250, n+2)[1:-1]
        y = x*0 + 250
        self.P = np.vstack((x, y)).T
        return self.P

    def find_ratio(self, P_in=None):
        """Returns the ratio of area to perimeter"""
        #if len(P.shape)==1:
        #    P = (np.insert(P, 1, 250)).reshape(int((len(P)+1)/2),2)
        if P_in is not None:
            P = (np.insert(P_in, 1, 250)).reshape(int((len(P_in)+1)/2),2)
            self.P = P
        self.find_area()
        self.find_lengths()
        self.ratio = self.area/self.perimeter
    
    def optimise(self):
        """optimises using scipy because fuck it.
        note that this function doesn't force x<=y. this is because it isn't
        necessary to make this bound explicit: it just does it itself. because
        x<=y isn't an explicit bound, the set of coordinates is an N dimensional
        square with side of {0, 250}, and N = 2*number of coordinates - 1.
        I don't know much about symmetry about hyperplanes, but I do know that
        N dimensional squares are always symmetric"""
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
        """Makes a new P by finding the sharpest corner and splitting it."""
        P = self.P
        self.P_old = P          # good to have just in case
        
        yi = np.sum(P[-1])/2    # coordinates of the symmetry intercept
        
        P_split = np.vstack((P, np.array([yi, yi])))
        #print(P_split)
        
        Xs, Ys = np.vstack((np.array([0, 250]), P_split)).T
        grads = (Ys[1:]-Ys[:-1])/(Xs[1:]-Xs[:-1])   # gradients of each line section
        thetas = np.arctan(grads)                   # angle of each line section with respect to x axis
        angles = np.pi - thetas[:-1] + thetas[1:]   # the angle between each line segment
        
        pos_max = angles.argmin()   # position of the sharpest corner
                                    # the index should line up if using P_split
        print(pos_max)
        #print(angles)
        new_X = np.zeros(len(P_split[1:-1]) + 1)    # i'm using P_split because if a coordinate is
        new_Y = np.zeros(len(P_split[1:-1]) + 1)    # too close to y=x, it won't integerise nicely
        
        
        if (pos_max <= (len(P_split)-3)):   # what the fuck is this? whoever wrote this should
            Xs = Xs[1:-1]                   # be ashaimed for not commenting probably
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
        I think the new P will be further from optimum than the new P made by
        splitting the sharpest corner, but it's more straightforward"""
        P = self.P
        i = np.argmax(self.lengths)
        if i == 0:
            new_C = P[0]/2
        else:
            new_C = (P[i-1]+P[i])/2
        self.P = np.vstack((P[:i], new_C, P[i:]))
    
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
        """Finds the gradient of the area
        Todo: allows for positive dx, negative dy, and dx>dy, for coordinates
        on the wrong side of y=x. Might be a flaw in the math?"""
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
    
# =============================================================================
#     def lin_R(self, U):
#         """a linearised form of ratio(), suitable for linear MIP solvers. it
#         uses the currently stored value of P, which should be updated only after
#         each solve is completed. This should be an issue as a third-party solver
#         isn't going to touch self.P, but if it is, this function can be
#         reformulated as a class so that it can hold on to its own initial P
#             U is a flattened, updated form of P"""
#         R0 = self.ratio
#         dR = self.dR
#         P0 = self.flat_P
#         return R0 + np.sum((U-P0), dR)[0]
#         
#     def init_lin_R(self):
#         """initiates the variables used in lin_R. Must be called at the
#         beginning of the MIP cycle.
#         The variables are initiated in a seperate function so that they don't
#         get re-computed every time lin_R() is called
#         """
#         self.find_ratio()
#         self.dR = self.del_ratio()
#         self.flat_P = np.delete(self.P, 1)
# =============================================================================
        
    def MIP(self, tol=1):
        """maximises the ratio in integer space
        TODO: the function is cyclic, and trying to add the second derivative
        to the linearisation didn't help. The coordinates don't really move from
        the initial value, so the best alternative will be to enumerate all
        possible values and find the highest ratio, at a cost of
        3^(len(flat_P)-1) function calls.
            Other options are to enumerate between the two cycled values, use
        a different solver that accepts my quadratics, or get SCIP to work
        non-linear. Or read a bunch of journal papers and figure something else
        out.. Or just bypass cvxpy and go straight for a solver. or maybe re-do
        the problem so instead of integers, the variables have a resolution of
        .5, and use the answer as a base to start solving?"""
        # initialize the constants
        
        #self.P_old = self.P*0
        #np.delete(q.P.flatten(), 1)
        self.flat_P = np.delete(self.P.flatten(), 1)
        flat_P = self.flat_P
        len_P = len(flat_P)
        self.old_flat_P = flat_P*0
        #self.dR = self.del_ratio()
        
        res_buffer = (np.zeros(len_P),)*4   # FIFO buffer of the results
        
        # create objective
        R0 = cvx.Parameter()    # Ratio at the initial point
        dR = cvx.Parameter(len_P)   # del Ratio
        P0 = cvx.Parameter(len_P)   # initial point of each iteration
        U = cvx.Variable(len_P, integer=True)   # iterative variable ie. x0+h
        #H = cvx.Parameter((len_P, len_P))
        
        """linearised ratio is cyclic. each iteration is clearly over-shooting
        the optimum, and I think that's because R0 changes with each iteration.
        It must change because it needs to be in the same place as dR, which
        obviously is 0. The options for improvement are therefore: 1. dampen dR
        until the point where the oscillations stop and both initial coordinates
        agree on the end coordinates; 2. include a the next term in the taylor
        series (tried that, cvxpy didn't like it); or 3. unpack the ratio before
        linearising it. Option 3. is similair to Dinkelbach’s Transform, except
        the optimal auxillary variable is already known"""
        #objective = cvx.Maximize(R0 + (U-P0).T * dR) # + (U-P0).T*H*(U-P0))
        
        """unpacked ratio, that is then linearised"""
        R_opt = self.ratio      # the floating-point ratio i.e. non-integer solution
        A0 = cvx.Parameter()
        L0 = cvx.Parameter()
        dA = cvx.Parameter(len_P)
        dL = cvx.Parameter(len_P)
        objective = cvx.Maximize(A0 + (U-P0).T * dA - R_opt*(L0 + (U-P0).T * dL))
        
        # constraints
        bl = cvx.Parameter(len_P)
        bu = cvx.Parameter(len_P)
        constraints = [bl <= U, U <= bu, U <= 250, U >= 0]
        
        while (self.flat_P != self.old_flat_P).all():
            self.old_flat_P = self.flat_P
            flat_P = self.flat_P
            #print(flat_P)
            # set values
            self.find_ratio(flat_P)
            
# =============================================================================
#             # values for directly linearised ratio
#             R0.value = float(self.ratio)    # Ratio at the initial point
#             J = self.del_ratio(flat_P)
#             dR.value = J                    # del Ratio
# =============================================================================
            
            # values for unpacked ratio
            A0.value = self.area
            L0.value = self.perimeter
            dA.value = self.del_area()
            dL.value = self.del_lengths()
            
            P0.value = self.flat_P          # initial point of each iteration
            bl.value = flat_P-tol           # lower bounds of the problem
            bu.value = flat_P+tol           # upper bounds of the problem
            
            #H.value = np.dot(np.vstack((J, np.zeros((2, 3)))).T, np.vstack((J, np.zeros((2, 3)))))
            
            #print(U.value, flat_P)
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.GLPK_MI)
            #prob.solve(solver=cvx.ECOS_BB)
            #print(R0.value, prob.value, U.value, flat_P)
            print(prob.value, U.value, flat_P)
            self.flat_P = U.value
            self.MIP_solve = prob
            
            # results buffer update and oscillation check
            res_buffer = res_buffer[1:] + (flat_P,)
            if (res_buffer[0] == res_buffer[2]) and (res_buffer[1] == res_buffer[3]):
                break
        
        def shuffle(self):
            """shuffles the coordinates up and down by 1 until and optimum is
            found. This might be needed because I'm not 100% sure that the set
            is symmetric, but I am sure that it's a cross-shaped set because the
            problem is a 2d shadow inside the values set, so it's pretty easy
            to just how the potential values connect together. If I'm wrong,
            I'll make this function, but only if I'm wrong.
            the method I'd use is to convert an integer to a binary string, add
            zeros to the front, and convert the string to a numpy array,
            ie. something like np.array(list(bin(i)[2:]), dtype=int) where i is
            the iterative"""
            
        



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
    
    proj.find_lengths()
    del_l = np.round(proj.del_lengths(), 8)
    del_a = np.round(proj.del_area(), 8)
    del_r = np.round(proj.del_ratio(), 8)
    
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
    