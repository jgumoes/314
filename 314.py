# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:43:04 2017

@author: Jack

2nd attempt at project euler question 314
"""

import pyqtgraph as pg
import time

app = pg.mkQApp()
plt = pg.plot()
plt.setYRange(0, 500)
plt.setXRange(0, 500)


import random

def update():
    x = []
    y = []
    for i in range(500):
        x.append(random.randrange(500))
        y.append(random.randrange(500))
    plt.plot(x, y, clear=True)
    #time.sleep(1)

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000)