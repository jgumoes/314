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
print("\n")
print("################")

shuffle = ctypes.CDLL("C:\\Users\\jgumo\\Documents\\Scripts\\Python Scripts\\project euler\\314\\C code\\c_shuffle.dll")

shuffle.find_ratio_flat.restype = ctypes.c_double # this corrects the ratio when function doesn't accept arguments and also when it does
shuffle.find_ratio_flat.argtypes = [ctypes.c_void_p, ctypes.c_int] # this gets rid of error when function accepts arguments, fixes everything, might cure aids but haven't gotten that far yet
p_9 = (np.array([123, 135, 146, 158, 168, 178, 188, 198, 207, 249, 247, 244, 240, 235, 230, 223, 215])).ctypes.data
ratio = shuffle.find_ratio_flat(p_9, 9) # *used to* returns ctypes.ArgumentError: argument 1: OverflowError: int too long to convert

#ratio = shuffle.find_ratio_flat()   # there *was* a stack overflow when the c function takes arguments and non are given,

print(type(ratio))  # type *was* int, despite function returning double
print(ratio)        # ratio *was* returned either as a pointer, or butchered double
print("true ratio = 132.49949177788744")

# scoot_C and scoot_np both work fine. Below was where I had mis-typed, but it still worked.
# however, entering the array to scoot_C will mutate the array. still, good to know.
shuffle.scoot_np.restype = ctypes.c_double
shuffle.scoot_np.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
print(shuffle.scoot_np(p_9, 18, 9))
print(p_9)
print(shuffle.scoot_np(p_9, 18, 9))
print(p_9)