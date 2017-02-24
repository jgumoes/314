# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 18:01:41 2017

@author: Jack

project euler question 314
https://projecteuler.net/problem=314
"""

import matplotlib.pyplot as py
import numpy as np

def first_try():
    """First attempt at the question"""
    x = np.linspace(0, 250, 251)
    #y = np.sqrt(250**2 - x**2)
    y = np.array([])
    for i in x:
        y = np.append(y, round(np.sqrt(250**2 - i**2)))
    
    area = np.sum(y[1:]) + (y[0]-y[-1])/2
    perimeter = 0
    
    count = 0
    
    while count < len(x) -1:
        perimeter += np.sqrt(1 + (y[count+1] - y[count])**2)
        count += 1
    
    py.scatter(x, y)
    py.show()

def cut_corners():
    """optimize for a square with cut corners.
    optimized cut length l = 77"""
    l = np.linspace(0, 250, 251)
    d = 500
    y = (d**2 - 2* l**2)/(4*d +(4*np.sqrt(2)-8)*l)
    py.plot(l, y)
    py.show()
    i = np.argmax(y)
    return l[i], round(y[i], 5)

def double_cut_corners():
    """optimize for a square with corners double cut"""
    ls = np.linspace(0, 250, 251)
    m = np.linspace(0, 250, 251)
    for l in ls:
        py.plot(((500**2)/8. - l*m/(2*np.sqrt(2)))/(500/2. + np.sqrt(l**2 + m**2 - l*m*np.sqrt(2))))
    py.show()