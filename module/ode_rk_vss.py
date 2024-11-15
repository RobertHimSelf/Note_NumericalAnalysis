#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable

try:
    from .ode__typing import X,Y
except:
    from ode__typing import X,Y

#y'=f(x,y)

'''显式变步长 Runge-Kutta 方法

通过一个四阶一个五阶RK方法的结果做差，估计四阶方法的局部误差。

然后令 newh = 0.9*h*(max_error/error)**(1/5) 使局部截断误差在 max_error 范围内。
依旧具有四阶精度，但总的误差更小。
'''

RungeKuttaFehlberg45 = [
    [
        [1/4, [1/4]],
        [3/8, [3/32, 9/32]],
        [12/13, [1932/2197, -7200/2197, 7296/2197]],
        [1, [439/216, -8, 3680/513, -845/4104]],
        [1/2, [-8/27, 2, -3544/2565, 1859/4104, -11/40]]
    ],
    [25/216, 0, 1408/2565, 2197/4104, 1/5, 0],
    [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
]
RKF = RungeKuttaFehlberg45

DormandPrince45 = [
    [
        [1/5, [1/5]],
        [3/10, [3/40, 9/40]],
        [4/5, [44/45, -56/15, 32/9]],
        [8/9, [19372/6561, -25360/2187, 64448/6561, -212/729]],
        [1, [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]]
    ],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40] #1/40对应的是 f(x+h, newy) 那项，这里姑且认为不影响，所以不处理。
]
DP = DormandPrince45

def _step(x, y, h, f, max_error, arg):
    hk = [h*f(x,y),]
    for i in range(5):
        newx = x+h*arg[0][i][0]
        newy = y
        for j in range(i+1):
            newy += arg[0][i][1][j]*hk[j]
        hk.append(h*f(newx,newy))
    newy = y + sum([arg[1][i]*hk[i] for i in range(6)])
    newy_r5 = y + sum([arg[2][i]*hk[i] for i in range(6)])
    error = abs(newy_r5-newy)
    if error < max_error:
        return x+h, newy
    h = 0.9*h/(error/max_error/newy)**(1/5)
    hk = [h*f(x,y),]
    for i in range(5):
        newx = x+h*arg[0][i][0]
        newy = y
        for j in range(i+1):
            newy += arg[0][i][1][j]*hk[j]
        hk.append(h*f(newx,newy))
    newy = y + sum([arg[1][i]*hk[i] for i in range(6)])
    return x+h, newy

def VariableStepSize_RungeKutta(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y,
    init_h:Number, max_x:Number,
    max_error:Number,
    arg=RKF):
    x=[x_0,]; y=[y_0,]
    while x[-1] < max_x:
        newx, newy = _step(x[-1],y[-1],init_h,f,max_error,arg)
        x.append(newx)
        y.append(newy)
    return (x,y)

if __name__ == "__main__":
    pass    #预留，不做任何处理