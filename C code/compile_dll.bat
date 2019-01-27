@echo off
title Compile DLL
gcc -fPIC -shared -o c_functions.dll c_functions.c
pause