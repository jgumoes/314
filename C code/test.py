"""
A small testing script to test the functions I make in c_functions.

There isn't a simple way to unload a dll except for closing to kernal
and re-doing everything, which means small adjustments to c_functions
is a huge fath. This script is makes life easier (if it's run from
command prompt. Loading it to an existing python kernal defeats the
whole point of it.)
"""

import ctypes
import numpy as np

cfunc = ctypes.CDLL("C:\\Users\\jgumo\\Documents\\Scripts\\Python Scripts\\project euler\\314\\C code\\c_functions.dll")
arr = np.linspace(0, 20, 21, dtype="int")
print(bin(5))

#print(cfunc.power(4))
#print(cfunc.butts())
#print(cfunc.sum(ctypes.c_void_p(arr.ctypes.data), len(arr)))
#print(cfunc.array())
print(cfunc.bin0(5, 11))
cfunc.print_bin()