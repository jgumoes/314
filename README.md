# 314
The repository dedicated to Project Euler question 314.

## What is question 314?
The question is to find the optimum area/perimeter ratio for a shape inside of a 500x500 square, with the constraint of being in integer space i.e. all coordinates are integers.

## First attempt
My very first attempt (which isn't in the repo) was to find the sharpest corner on the shape, squash it down, and repeat until the optimum is reached. Needless to say, this is over-complicated and a fantastic way to get over-whelmed.

## Real attempt
The final attempt was inspired when I was reading into using Tikhonov regularization for smooth differentiation of experimental data. The approach I've taken is to define coordinates along the edge of the shape, and use simple optimisation routines to find the optimal location for each coordinate. Once that has been found, another coordinate is added to the shape and all the coordinates are optimised again. As coordinates are added and re-optimised, the ratio increases as the perimeter is better defined. Because each coordinate must be an integer, it is possible to over-define the perimeter, causing the ratio to decrease with each added coordinate. This bilevel approach might be turned into a single-stage routine if the equation for the shape in real (non-integer) space is known, and that might be a fun project for the future, but the speed isn't an issue for this code. If I wanted to improve the speed some more, it would probably be better worth defining an analytical Hessian matrix.
