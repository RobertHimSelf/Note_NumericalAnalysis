#!/usr/bin/python
# -*- coding: utf-8 -*-

'''解非线性方程的二分法'''

import numpy as np
from numbers import Number
from typing import Callable

try:
    from ._matfunc import *
    from .iter_condition import StopCondition, stopAt
    from .ode__typing import X, Y, X0, Xmin, Xmax
except:
    from _matfunc import *
    from iter_condition import StopCondition, stopAt
    from ode__typing import X, Y, X0, Xmin, Xmax

def dichotomy(
    f:Callable[[X],Y],
    xmin:Xmin,
    xmax:Xmax,
    stop:StopCondition = stopAt(),
    showlog:bool = False
    ) -> X:
    '''二分法 / dichotomy
    解 f(x) = 0 的方法
    要求 f 是连续函数，f(xmin)*f(xmax)<=0'''
    if (fxmin:=f(xmin)) == 0: return xmin
    if (fxmax:=f(xmax)) == 0: return xmax
    if (fxmin < 0 and fxmax < 0) or (fxmin > 0 and fxmax > 0):
        raise ValueError("需要f(xmin)*f(xmax)<=0")
    
    if fxmin < 0: k = -1
    else: k = 1
    del fxmin, fxmax

    time = 0
    if showlog: print(f"开始二分法，初始区间为[{xmin}, {xmax}]")

    while not stop(xmin, xmax, time):
        newx = (xmin+xmax)/2
        if (fnewx:=f(newx)) == 0:
            if showlog: print(f"在迭代过程中找到解{newx}")
            return newx
        elif fnewx*k > 0:
            xmin = newx
        else:
            xmax = newx
        time += 1
        if showlog: print(f"第{time}步区间：[{xmin}, {xmax}]")
    return (xmin+xmax)/2

def false_position(
    f:Callable[[X],Y],
    xmin:Xmin,
    xmax:Xmax,
    stop:StopCondition = stopAt(),
    showlog:bool = False
    ) -> X:
    '''试位法 / regula falsi method / false position method
    解 f(x) = 0 的方法
    要求 f 是连续函数，f(xmin)*f(xmax)<=0'''
    if (fxmin:=f(xmin)) == 0: return xmin
    if (fxmax:=f(xmax)) == 0: return xmax
    if (fxmin < 0 and fxmax < 0) or (fxmin > 0 and fxmax > 0):
        raise ValueError("需要f(xmin)*f(xmax)<=0")
    
    if fxmin < 0: k = -1
    else: k = 1

    time = 0
    if showlog: print(f"开始试位法，初始区间为[{xmin}, {xmax}]")

    while not stop(xmin, xmax, time):
        newx = (xmax*fxmin-xmin*fxmax)/(fxmin-fxmax)
        if (fnewx:=f(newx)) == 0:
            if showlog: print(f"在迭代过程中找到解{newx}")
            return newx
        elif fnewx*k > 0:  #对于凸函数，在迭代过程中可以直接得到f(newx) <= 0，从而可以减少一个判断条件
            xmin = newx
            fxmin = fnewx
        else:
            xmax = newx
            fxmax = fnewx
        time += 1
        if showlog: print(f"第{time}步区间：[{xmin}, {xmax}]")
    return (xmin+xmax)/2