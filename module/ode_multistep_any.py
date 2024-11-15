#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable
from fractions import Fraction

try: from .ode__typing import X,Y
except: from ode__typing import X,Y

#y'=f(x,y)

def Gauss_origin(a:list[list[Fraction]], b:list[Fraction]) -> list[Fraction]:
    '''经典高斯消去法，不进行任何高级操作
    这里不需要高级操作的原因是已经使用了无误差的数据类型，且确保了主元非零'''
    n = len(a)
    for i in range(n):            #行变换形成上三角矩阵
        for j in range(i+1, n):
            b[j] -= b[i]*(temp:=a[j][i]/a[i][i])
            for k in range(n):
                a[j][k] -= a[i][k]*(temp)
    for i in range(n-1, -1, -1):  #行变换形成单位矩阵
        b[i] /= a[i][i]
        a[i][i] = 1
        for j in range(i):
            temp = a[j][i]
            b[j] -= b[i]*temp
            for k in range(n):
                a[j][k] -= a[i][k]*temp
    return b

def getweight(rank:int):
    a = [[Fraction((-i)**j) for i in range(rank)] for j in range(rank)]
    b = [Fraction(1,k) for k in range(1,rank+1)]
    return Gauss_origin(a,b)

def multistep(
    x_init:list[X],
    y_init:list[Y],
    f:Callable[[X, Y], Number], 
    max_x:Number, 
    h:Number,
    rank:int
    ) -> tuple[list[X],list[Y]]:
    '''rank 阶多点法'''
    x=[x_i for x_i in x_init] 
    y=[y_i for y_i in y_init]

    weight = getweight(rank)
    k = len(weight)

    #'''
    dominator_LCM = 1
    for i in range(k):
        temp = weight[i].denominator
        dominator_LCM *= temp
        for j in range(k):
            weight[j]*=temp
    for i in range(k):
        weight[i] = int(weight[i])
    divided_h = h/dominator_LCM
    '''
    divided_h = h
    #'''
    
    dy = [0] + [f(x[i],y[i]) for i in range(-k,-1,1)]
    
    while x[-1] < max_x:
        for i in range(len(dy)-1):
            dy[i] = dy[i+1]
        dy[-1] = f(x[-1], y[-1])
        y.append(y[-1]+divided_h*sum([dy[-i-1]*weight[i] for i in range(k)]))
        x.append(x[-1]+h)
    return x,y

def test(f, x0, max_x, h,  g, rank, func):
    #y = g(x)
    #y'= f(x,y)
    #x,y = multistep(...)
    y0 = [g(x) for x in x0]
    x,y = func(x0, y0, f, max_x, h, rank)
    error = 0
    for i in range(len(x)):
        if error<(newerror:=abs(y[i]-g(x[i]))):
            error = newerror
    print(error)
    '''
    with d.localcontext() as ctx:
        ctx.prec = 50
        print(d.Decimal(error)) #'''

if __name__ == "__main__":
    import math
    h = 0.01
    x0 = [i*h for i in range(20)]
    max_x = 10
    f = lambda x,y: x-y+1
    g = lambda x: math.exp(-x)+x
    #f = lambda x,y: math.exp(x)*y**3
    #g = lambda x: math.sqrt(0.5/(30000-math.exp(x)))
    #f = lambda x,y: (1-math.sin(x))*y**4
    #g = lambda x: -(3*(x+math.cos(x)-0.999999))**(-1/3)

    max_rank = 20 
    for i in range(1,max_rank):
        #print(repr(getweight(i)))
        #pass
        #'''
        print(f"{i}阶多步法误差为", end="")
        try:
            test(f, x0, max_x, h, g, i, multistep)
        except OverflowError as e:
            print("OverflowError")
        #'''
    #'''



    