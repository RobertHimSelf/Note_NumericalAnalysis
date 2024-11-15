#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable

try:
    from .ode__typing import X,Y
except:
    from ode__typing import X,Y

#y'=f(x,y)

'''显式 Runge-Kutta 方法
显式方法, 计算量更小, 效率更高

显式 RK 方法可以概括为：
a=(a_1, a_2, ..., a_r)^T, a_1 = 0
B=(b_i,j) 为主对角线为零的下半矩阵
k=(k_1, k_2, ..., k_r)^T
k = f(x+h*a, y+B*k) 由于B是主对角线为零的下半矩阵, 可以逐行递推得到显式方法

然后作一个权重向量c = (c_1, ..., c_r), sum(c)=1, 取：
y[n+1] = y[n] + h*c*k 即完成计算

计算精度的阶数决定了对 a, B, c 中各分量的取值要求。下面列举的参数要求即为取值要求。

隐式 RK 方法类似，但不要求 a_1 = 0, 也不要求 B 为主对角线为零的下半矩阵
此时用 k = f(x+h*a, y+B*k) 进行迭代
'''

def r2RungeKutta(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y, 
    h:Number, max_x:Number,
    c1=0.0, c2=1.0, a2=0.5, b21=0.5
    ) -> tuple[list[X],list[Y]]:
    '''二阶Runge-Kutta法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h，精度 O(h^2)
    参数要求：c1+c2==1, a2*c2==0.5, b21*c2==0.5

    如果未指定参数，用默认输入，这是中点法（或者称为显式梯形法）
    '''
    x=[x_0,]; y=[y_0,]
    while x[-1] < max_x:
        k1 = f(x[-1],y[-1])
        k2 = f(x[-1]+h*a2, y[-1]+h*b21*k1)
        x.append(x[-1]+h)
        y.append(y[-1]+h*(c1*k1+c2*k2))
    return (x,y)
def Midpoint(f,x,y,h,max_x):
    return r2RungeKutta(f,x,y,h,max_x)
def EulerModified(f,x,y,h,max_x):
    return r2RungeKutta(f,x,y,h,max_x,c1=0.5,c2=0.5,a2=1,b21=1)
def Heun(f,x,y,h,max_x):                              #Heun方法
    return r2RungeKutta(f,x,y,h,max_x,c1=0.25,c2=0.75,a2=2/3,b21=2/3)
r2Heun = Heun #Name Alias: 二阶Heun方法

#三阶Runge-Kutta法
def r3RungeKutta(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y, 
    h:Number, max_x:Number,
    c1=1/6, c2=2/3, c3=1/6,
    a2=0.5, b21=0.5,
    a3=1.0, b31=-1.0, b32=2.0):
    '''三阶Runge-Kutta法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h，精度 O(h^2)
    参数要求：c1+c2+c3==1, a2==b21, a3==b31+b32
    a2*c2+a3*c3==0.5, c2*a2**2+c3*a3**2==1/3
    c3*a2*b32=1/6
    如果未指定参数，用默认输入，这是Kutta法
    '''
    x=[x_0,]; y=[y_0,]
    while x[-1] < max_x:
        k1 = f(x[-1],y[-1])
        k2 = f(x[-1]+h*a2, y[-1]+h*b21*k1)
        k3 = f(x[-1]+h*a3, y[-1]+h*(b31*k1+b32*k2))
        x.append(x[-1]+h)
        y.append(y[-1]+h*(c1*k1+c2*k2+c3*k3))
    return (x,y)
def Kutta(f,x,y,h,max_x):                             #Kutta法
    return r3RungeKutta(f,x,y,h,max_x)
def r3Heun(f,x,y,h,max_x):                            #三阶Heun方法
    return r3RungeKutta(f,x,y,h,max_x,c1=0.25,c2=0,c3=0.75,a2=1/3,b21=1/3,a3=2/3,b31=0,b32=2/3)

#四阶Runge-Kutta法
def r4RungeKutta(
    f:Callable[[X, Y], Number], 
    x_0:X, y_0:Y,
    h:Number, max_x:Number,
    c1=1/6, c2=1/3, c3=1/3, c4=1/6,
    a2=0.5, b21=0.5,
    a3=0.5, b31=0.0, b32=0.5,
    a4=1.0, b41=0.0, b42=0.0, b43=1.0):
    '''四阶Runge-Kutta法
    用于微分方程初值问题 y'=f(x,y); y(x_0)=y_0 的方法
    迭代至 x >= max_x 为止，每次步长为 h，精度 O(h^2)
    参数要求：c1+c2+c3+c4==1, a2==b21, a3==b31+b32, a4=b41+b42+b43
    a2*c2+a3*c3+a4*c4==0.5, c2*a2**2+c3*a3**2+c4*a4**2==1/3
    c3*a2*b32+c4*a3*b43+c4*a2*b42=1/6
    参数要求未完待续

    如果未指定参数，用默认输入，这是经典Runge-Kutta法
    '''
    x=[x_0,]; y=[y_0,]
    while x[-1] < max_x:
        k1 = f(x[-1],y[-1])
        k2 = f(x[-1]+h*a2, y[-1]+h*b21*k1)
        k3 = f(x[-1]+h*a3, y[-1]+h*(b31*k1+b32*k2))
        k4 = f(x[-1]+h*a4, y[-1]+h*(b41*k1+b42*k2+b43*k3))
        x.append(x[-1]+h)
        y.append(y[-1]+h*(c1*k1+c2*k2+c3*k3+c4*k4))
    return (x,y)
def classicalRungeKuta(f,x,y,h,max_x):                #经典Runge-Kutta法
    return r4RungeKutta(f,x,y,h,max_x)

if __name__ == "__main__":
    pass    #预留，不做任何处理