#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable

try:
    from .ode__typing import *
    from .iter_condition import StopCondition, stopAt
except:
    from ode__typing import *
    from iter_condition import StopCondition, stopAt

'''线性多步法

注意到
(y[n+1] - y[n])/h = y' + 1/2 * hy" + 1/3 * hhy"'/2 + ...

y'[n-1] = y' -  hy" +     hhy"'/2 - ...
y'[n-2] = y' - 2hy" + 2^2 hhy"'/2 - ...
...
y'[n-k] = y' - khy" + k^2 hhy"'/2 - ...

通过解线性方程组，可以用下面的配凑出上面的。从而得到多步方法。

多步方法通常要先用其他方法达成足够多的步数，然后再使用。

除上述 y'[n-k] 之外，还可以用其他已知项，比如 y[n-1]、y[n-2] 等参与配凑；

也可以用 y'[n+1] 等未知项参与配凑，但此时方法会变为隐式方法
'''

#y'=f(x,y)
def duostep(
    x_init:tuple[X0, X1]|list[X],
    y_init:tuple[Y0, Y1]|list[Y],
    f:Callable[[X, Y], Number], 
    max_x:Number, 
    h:Number
    ) -> tuple[list[X],list[Y]]:
    '''二步法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h，精度 O(h^2)
    
    输入的 x_init 和 y_init 都是列表，列表中至少要有 2 个元素。
    要求y(x[n]])=y[n]是上述初值问题的解。
    通常情况下，会先用一个单步法得到 y(x[1]) = y[1]，再使用此方法
    
    注意到这个函数的名字是“二步舞”，我对这个命名很满意'''
    x=[x_i for x_i in x_init] 
    y=[y_i for y_i in y_init]
    dy = [0, f(x[-2],y[-2])]
    h2 = h/2
    while x[-1] < max_x:
        for i in range(len(dy)-1):
            dy[i] = dy[i+1]
        dy[-1] = f(x[-1], y[-1])

        y.append(y[-1]+h2*(3*dy[-1]-dy[-2]))
        x.append(x[-1]+h)
    return x,y

def Adams(
    x_init:tuple[X0, X1, X2, X3]|list[X],
    y_init:tuple[Y0, Y1, Y2, Y3]|list[Y],
    f:Callable[[X, Y], Number], 
    max_x:Number, 
    h:Number
    ) -> tuple[list[X],list[Y]]:
    '''Adams外推公式
    这是一个四步法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h，精度 O(h^4)
    
    输入的 x_init 和 y_init 都是列表，列表中至少要有 4 个元素。
    要求y(x[n]])=y[n]是上述初值问题的解。
    通常情况下，会先用其他方法得到四个初值点，再使用此方法
    (常用四阶Runge-Kutta法)'''
    x=[x_i for x_i in x_init] 
    y=[y_i for y_i in y_init]
    dy = [0] + [f(x[i],y[i]) for i in range(-4,-1,1)]
    h24 = h/24
    while x[-1] < max_x:
        for i in range(len(dy)-1):
            dy[i] = dy[i+1]
        dy[-1] = f(x[-1], y[-1])

        y.append(y[-1]+h24*(55*dy[-1]-59*dy[-2]+37*dy[-3]-9*dy[-4]))
        x.append(x[-1]+h)
    return x,y

def AdamsImplicit(
    x_init:tuple[X0, X1, X2]|list[X],
    y_init:tuple[Y0, Y1, Y2]|list[Y],
    f:Callable[[X, Y], Number], 
    max_x:Number, 
    h:Number,
    stop:StopCondition = stopAt(e=0, rel_e=1e-10, max_iter=1000)
    ) -> tuple[list[X],list[Y]]:
    '''Adams内插公式
    这是一个四步法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h，精度 O(h^4)
    
    输入的 x_init 和 y_init 都是列表，列表中至少要有 3 个元素。
    四步法只要三个元素，是因为有一步是隐式存在的，所以只需要三个初值点
    要求y(x[n]])=y[n]是上述初值问题的解。
    通常情况下，会先用其他方法得到三个初值点，再使用此方法
    (常用四阶Runge-Kutta法)'''
    x=[x_i for x_i in x_init] 
    y=[y_i for y_i in y_init]
    dy = [f(x[i],y[i]) for i in range(-3,-1,1)]
    h24 = h/24
    while x[-1] < max_x:
        x.append(x[-1]+h)
        newy = y[-1]; y.append(0)
        
        times = 0; temp = h24*(19*dy[-1] - 5*dy[-2] + dy[-3])
        while not stop(y[-1], newy, times):
            y[-1] = newy
            newy = y[-2] + temp + h24*(9*f(x[-1], y[-1]))
            times += 1
        y[-1] = newy
        for i in range(len(dy)-1):
            dy[i] = dy[i+1]
        dy[-1] = f(x[-1], y[-1])
        
    return x,y

if __name__ == "__main__":
    pass    #预留，不做任何处理