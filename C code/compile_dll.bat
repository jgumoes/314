@echo off
title Compile DLL
gcc -fPIC -shared -o c_functions.dll c_functions.c
gcc -fPIC -shared -o c_shuffle.dll c_shuffle.c
pause