# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:43:04 2017

@author: Jack

2nd attempt at project euler question 314
"""

import pyqtgraph as pg
import numpy as np

app = pg.mkQApp()
plt = pg.plot()
plt.setYRange(0, 500)
plt.setXRange(0, 500)

# make a square
x = [1, 1, 500, 500]
y = [1, 500, 500, 1]
sq = np.matrix([x, y])
x, y = sq.getA()
plt.plot(x, y)


def update():
    global sq
    rot = np.matrix([[0, 1], [-1, 0]])
    sq = 250.5 + rot.dot((sq-250.5))
    x, y = sq.getA()
    plt.plot(x, y, clear=True)
    
    if not plt.isVisible():
        timer.stop()

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000)