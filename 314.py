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

# make a triangle
x = [200, 300, 400, 200, 200, 200]
y = [200, 200, 200, 400, 300, 200]
tri = np.matrix([x, y])
x, y = tri.getA()
plt.plot(x, y)


def update():
    global tri
    rot = np.matrix([[0, 1], [-1, 0]])
    tri = 250.5 + rot.dot((tri-250.5))
    x, y = tri.getA()
    plt.plot(x, y, clear=True)
    
    if not plt.isVisible():
        timer.stop()

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000)