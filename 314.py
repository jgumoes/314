# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:16:26 2018

@author: Jack

project euler problem 314
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
        self.optimise()
        
    
    def find_lengths(self):
        """outputs the perimeter, given coordinates P
        P0 is (x0, 250), and Pe is the last coordinate before the line of symmetry"""
        P = self.P
        L = np.zeros(len(P)+1) # L = lengths of each line
        L[0] = P[0][0]*8
        xe, ye = P[-1]
        L[-1] = np.sqrt((ye**2 + xe**2 - 2*ye*xe)*32)
        
        if len(P) != 1:
            dP = (P[1:] - P[:-1]).transpose()
            L[1:-1] = np.sqrt(64*(dP[0]**2 + dP[1]**2))
        
        self.lengths = L
        self.perimeter = np.sum(L)
    
    def find_area(self):
        """Finds the total area using the trapezium rule"""
        P = self.P
        A = np.zeros(len(P) + 1) # areas of each triangle segment
        
        yi = np.sum(P[-1])/2    # coordinates of the symmetry intercept
        P = np.vstack((P, np.array([yi, yi])))
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
        a hissy fit (is the jacobian is passed straight to .optimise(), it
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
        TODO: finish"""
    
    def del_lengths(self):
        """finds the gradient of the lengths
        TODO: make it work proper"""
# =============================================================================
#         if P_in is None:
#             P = self.P
#             P_in = np.zeros(len(P)*2 - 1)
#             P_in[0] = P[0, 0]
#             P_in[1:] = (P.flatten())[2:]
#         else:
#             P = (np.insert(P_in, 1, 250)).reshape(int((len(P_in)+1)/2),2)
#             self.P = P
#         if P_in is not None:
#             P = (np.insert(P_in, 1, 250)).reshape(int((len(P_in)+1)/2),2)
#             self.P = P
#         else:
#             P = self.P
#         self.find_lengths()
# =============================================================================
        lengths = self.lengths/8
        
        P = self.P
        
        Pi = np.sum(P[-1])/2
        fP = np.vstack((np.array([0, 250]), P, np.array([1, 1])*Pi))
# =============================================================================
#         LHS = np.zeros(2*len(P) -1)
#         RHS = np.zeros(2*len(P) -1)
#         LHS[0] = 1
#         ddP = P[1:] - P[:-1]    # double-delta-P makes sense in flattened form
#         print("ddP: {}".format(ddP))
#         ddP = ddP/np.vstack((lengths[1:-1], lengths[1:-1])).T
#         flat_ddP = ddP.flatten()
#         print("lengths: {}".format(np.vstack((lengths[1:-1], lengths[1:-1])).T))
#         print("flat_ddP: {}".format(flat_ddP))
#         #print(LHS, RHS, flat_ddP)
#         LHS[1:] = flat_ddP[0]
#         LHS[2:] = flat_ddP[2:-3]
#         
#         RHS[0] = flat_ddP[0]
#         RHS[1:-2] = flat_ddP[2:]
#         RHS[-2] = ((P_in[-2:][0] - P_in[-2:][1])/(2*lengths[-1]))/2
#         RHS[-1] = -RHS[-2]
#         #RHS = RHS * np.append(1, np.tile(np.array([1, -1]), int(len(RHS)/2)))
#         print("LHS: {} \n RHS: {}".format(LHS, RHS))
#         return 8*(LHS - RHS)
# =============================================================================
        ddP = fP[1:] - fP[:-1]
        ddP2 = ddP/np.vstack((lengths, lengths)).T
        dels = (ddP2[:-1] - ddP2[1:]).flatten()
        del_out = np.zeros(len(dels)-1)
        del_out[0] = dels[0]
        del_out[1:] = dels[2:]
        return del_out*8
    
    def del_area(self):
        """Finds the gradient of the area"""
        P = self.P
        Pi = np.sum(P[-1])/2
        fP = np.vstack((np.array([0, 250]), P, np.array([1, 1])*Pi))
        ddP = fP[2:] - fP[:-2]
        #print("ddP: {}".format(ddP*4))
        dels = 4*np.vstack((-ddP[:, 1], ddP[:, 0])).T
        #print("del_area: {}".format(4*np.vstack((-ddP[:, 1], ddP[:, 0]))))
        #print("flat del_area: {}".format((4*np.vstack((-ddP[:, 1], ddP[:, 0])).T).flatten()))
        dels_end = 4*np.array([fP[-3, 1]-fP[-2, 0], fP[-2, 1]-fP[-3, 0]])
        #dels_end = 4*np.array([P[-2, 1]-P[-1, 0], P[-1, 1]-P[-2, 0]])
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





result = None


def test_del_lengths(n=None, eps=10**-5):
    """a function that brute-forces the jacobian to test the math.`
    TODO: finish for all of the jacobian"""
    proj = q_314()
    if n is None:
        proj.P = np.array([[ 50., 250.], [100., 230.], [150., 215.], [200., 210.]])
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













def next_P(P):
    """finds the biggest corner, then splits it and optimizes again"""
    
    
    
# =============================================================================
# def iterate_opt(P):
#     """iterates opt() using the new coordinates found in next_P()
#     doesn't seem to be functional"""    
#     r = ratio(P)
#     print(r)
#     out_r = 0
#     P_next = P
#     while ((out_r < r) and (out_r != r)):
#         r = out_r
#         out = opt(make_P_init(len(P_next)+1))
#         out_p = (out.x).round() 
#         P_next = (np.insert(out_p, 1, 250)).reshape(int((len(out_p)+1)/2),2)
#         out_r = ratio(P_next)
#         print(len(P_next), out_r)
# =============================================================================

def it_opt(X):
    """scipy-friendly wrapper for opt()"""
    #P = (np.insert(X, 1, 250)).reshape(int((len(X)+1)/2),2)
    P = make_P_init(X)
    #opres = opt(P)
    global result
    result = opt(P)
    #print((len(result.x)+1)/2)
    rat = np.round(result.x)
    if rat[-1] >= rat[-2]:
        rat = rat[:-2]
    return np.round(ratio(rat), 8) #float(opres.fun)
    

def iterate(n=20):
    """optimises the optimisation"""
    return op.minimize(it_opt, n, options={"maxfun": 10**5, "eps": 1})
    

def plot(P):
    """Plots the quater graph made by P.
    If P is in the flattened form, it's reshaped to coordinate form"""
    if len(P.shape) == 1:
        P = (np.insert(P, 1, 250)).reshape(int((len(P)+1)/2),2)
    x, y = P.T
    graph = plt.figure()
    graph.line(np.hstack((x, y[::-1])), np.hstack((y, x[::-1])))
    plt.show(graph)
    
def fuck_it(stop=100, start=2):
    """fuck it, no optimisation, no iteration, this function will just do
    it_opt() for every single value between start and stop"""
    opt_res = []
    for i in range(start, stop+1):
        opt_res.append(it_opt(i))
        print(i)
    return opt_res

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
