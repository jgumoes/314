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
y = np.append(400*np.ones(100), 500*np.ones(150))
x = np.append(np.array(range(1, 501)), np.array(range(1, 501))[::-1])
x = np.append(x, 1)
plt.plot(y)

def mirror(coord):
    for i in range(50):
        y[i] = r.randint(251, 500)
    top = np.append(y, y[::-1])
    top = np.append(top, 501-top)
    top = np.append(top, top[0])
    return top

#plt.plot(x, mirror(y), clear=True)

import random as r

def update():
    for i in range(50):
        y[i] = r.randint(251, 500)
    top = np.append(y, y[::-1])
    top = np.append(top, 501-top)
    top = np.append(top, top[0])
    plt.plot(x, top, clear=True)
    if not plt.isVisible():
        timer.stop()

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(500)
