# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:43:04 2017

@author: Jack

2nd attempt at project euler question 314
"""

import pyqtgraph as pg
import numpy as np
import threading

app = pg.mkQApp()
plt = pg.plot()
plt.setYRange(1, 500)
plt.setXRange(1, 500)

# make the top sector
y = 500*np.ones(250)
x = np.append(np.array(range(1, 501)), np.array(range(1, 501))[::-1])
x = np.append(x, 1)
#plt.plot(y)

def diff(vals):
    """finds second order difference of vals.
    analogous to second-order differentiation"""
    return vals[0] + vals[2] - 2*vals[1]

def ratio(top, bot=None, end=False):
    """finds area/perimeter ratio between top and bottom lines"""
    if bot == None:
        bot = 501-top
    val = top-250.5 #bot
    i = 0
    area = (2*np.sum(val) - val[0] - val[-1])
    per = 0
    while i < len(val)-1:
        #area += (val[i]+val[i+1])/2
        per += np.sqrt(1+ (top[i+1]-top[i])**2.)
        i+=1
    per += val[0]
    #return area, per, area/per
    return area/per

def fudge(i=0):
    """fudges a point down (or up) to maximise it's ratio.
    i is the location of the corner"""
    #y_int = np.asfarray(y[0: i+2])
    global y
    # homebrew optimisation using increments instead of dumb scipy that i can't figure out
    inc = 1.  # the increment of the corner.
    while 501 > y[i] > 250:
        #print inc
        y_int = np.copy(y)
        y_int[i] -= inc # there isn't a better way of doing this
        if ratio(y) < ratio(y_int):
            y = y_int
        elif inc == 1:
            inc = -1.
        else:
            global last
            last = i
            break
last = 0
def run():
    while 1:
        i = 0
        last = 0
        while i < 250:
#            if i == last:
#                rotate()
            fudge(i)
            i+=1
        if not plt.isVisible():
            thread.stop()
            timer.stop
            

def update():
    top = np.append(y, y[::-1])
    top = np.append(top, 501-top)
    top = np.append(top, top[0])
    plt.plot(x, top, clear=True)
    if not plt.isVisible():
        timer.stop()
        thread.stop()

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

thread = threading.Thread(target=run)
thread.start()
