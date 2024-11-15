#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable

try:
    from .ode__typing import X,Y
    from .iter_condition import StopCondition, stopAt
except:
    from ode__typing import X,Y
    from iter_condition import StopCondition, stopAt

#y'=f(x,y)

'''欧拉折线法以及一些变种'''

def Euler(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y, 
    h:Number, max_x:Number
    ) -> tuple[list[X],list[Y]]:
    '''欧拉折线法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h
    精度 O(h)，稳定区间为 h*f(x,y)/y in [-2,0]'''
    x=[x_0]; y=[y_0]
    while x[-1] < max_x:
        y.append(y[-1]+h*f(x[-1],y[-1]))
        x.append(x[-1]+h)
    return x,y

def EulerImplicit(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y, 
    h:Number, max_x:Number, 
    stop:StopCondition = stopAt(e=0, rel_e=1e-10, max_iter=1000)
    ) -> tuple[list[X],list[Y]]:
    '''隐式欧拉折线法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h
    精度 O(h)，稳定区间为 h*f(x,y)/y < 0
    每次迭代需要进行子迭代：y[n+1]=y[n]+h*f(x[n+1],y[n+1])'''
    x=[x_0]; y=[y_0]
    while x[-1] < max_x:
        x.append(x[-1]+h)
        newy = y[-1]; y.append(0)
        times = 0
        while not stop(y[-1], newy, times):
            y[-1] = newy
            newy = y[-2] + h*f(x[-1],y[-1])
            times += 1
        y[-1] = newy
    return x,y

def EulerImproved(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y, 
    h:Number, max_x:Number, 
    stop:StopCondition = stopAt(e=0, rel_e=1e-10, max_iter=1000)
    ) -> tuple[list[X],list[Y]]:
    '''改进欧拉折线法（梯形方法）
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h
    精度 O(h^2)，稳定区间为 h*f(x,y)/y < 0
    每次迭代需要进行子迭代：y[n+1]=y[n]+h/2*(f(x[n+1],y[n+1])+f(x[n],y[n]))'''
    x=[x_0]; y=[y_0]; h2 = h/2
    while x[-1] < max_x:
        x.append(x[-1]+h)
        newy = y[-1]; y.append(0)
        times = 0; temp = f(x[-2],y[-2])
        while not stop(y[-1], newy, times):
            y[-1] = newy
            newy = y[-2] + h2*(f(x[-1],y[-1])+temp)
            times += 1
        y[-1] = newy
    return x,y

if __name__ == "__main__":
    pass    #预留，不做任何处理