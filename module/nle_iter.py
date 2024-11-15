#!/usr/bin/python
# -*- coding: utf-8 -*-

'''解方程的不动点迭代法'''

from typing import Callable
import math

try:
    from ._matfunc import *
    from .iter_condition import StopCondition, stopAt
    from .ode__typing import X, Y, X0, X1, X2
except:
    from _matfunc import *
    from iter_condition import StopCondition, stopAt
    from ode__typing import X, Y, X0, X1, X2

#从方程到迭代法
# 对于 f(x) = 0，通过变形得到 x = phi(x)
# 则 x[k+1] = phi(x[k]) 就是解方程的迭代法

#最简单的变形方式比如
# f(x) = 0
# alpha*f(x) = 0   (alpha != 0)
# x + alpha*f(x) = x
#取迭代法
# x = x + alpha*f(x)
#即可。为了迭代法的收敛性，可能对 alpha 的取值有所要求。

def fpi(
    phi: Callable[[X],Y],
    x0: X0 = 0,
    stop: StopCondition = stopAt(),
    showlog: bool = False) -> X:
    '''Fixed point iteration 不动点迭代法'''
    x = x0; time = 0
    if showlog:
        print(f"开始迭代，初值为：{x}")
    while not stop(x, x0, time):
        x0 = x
        x = phi(x0)
        time += 1
        if showlog: print(f"第{time}步迭代结果：{x}")
    return x

#对于迭代不动点 z 的邻域 U。对任意初值 x0 属于 U，取
# 迭代法得到的迭代序列为 x = [x0, ...]
# 若 x[k] - z 的极限为零，则称迭代法在邻域内局部收敛

#若迭代法局部收敛，且
# (x[k+1] - z)/(x[k] - z)**p 的极限为一个常数，则称迭代法在邻域内局部 p 阶收敛
# 特别的，p = 1 时称之为线性收敛，p = 2 时称之为平方收敛

#对于迭代法 x[k+1] = phi(x[k])
# 若 phi 在 U 上可导，且导数绝对值小于 1，则迭代法在 U 上局部收敛
# 若 phi 的导数在 U 上连续非零，则迭代法在 U 上线性收敛
#  可以证明极限为 phi 在 z 处的导数值
# 若 phi 的导数、2 阶导数、...、p-1 阶导数在 U 上连续，在 z 处取值为零
#  且 p 阶导数在 U 上连续非零，则迭代法在 U 上 p 阶收敛
#  可以证明极限为 phi 在 z 处的 p 阶导数值除以 p! 的结果

def Steffensen_Aitken(
    phi: Callable[[X],Y],
    x0: X0 = 0,
    stop: StopCondition = stopAt(),
    showlog: bool = False) -> X:
    '''使用 Aitken 加速方法的 Steffensen 迭代法

    要求 phi 的导数在邻域内变化足够小。
    
    Aitken 加速方法：
    注意到不动点迭代中
     x[k+1] - z = phi(x[k]) - phi(z)
    为 phi 的导数从 z 到 x[k] 的积分。若 phi 的导数在邻域内变化不大、都约等于 a，则
     phi(x[k]) - phi(z) 约等于 a * (x[k] - z)
    类似
     x[k+2] - z 约等于 a * (x[k+1] - z)
    从而
     (x[k+2] - z)/(x[k+1] - z) 约等于 (x[k+1] - z)/(x[k] - z)
    变形，得
     z 约等于 x[k] - (x[k+1] - x[k])**2 / (x[k+2] - 2*x[k+1] + x[k])
    把右侧估计结果作为迭代结果，就是 Aitken 加速方法。
    '''
    x = x0; time = 0
    if showlog: print(f"开始迭代，初值为：{x}")
    while not stop(x, x0, time):
        x0 = x
        x1 = phi(x0)
        x2 = phi(x1)
        x = x0 - (x1 - x0)**2/(x2 - 2*x1 + x0)
        if showlog: print(f"第{time}步迭代结果：{x}")
        time += 1
    return x

def Newton(
    f:Callable[[X],Y],
    x0:X0=0,
    df:Callable[[X],Y] = None,
    stop:StopCondition = stopAt(),
    showlog: bool = False) -> X:
    '''解 f(x) = 0 的牛顿迭代法
    df 为 f 的导数，如果 df 为 None，则使用默认的数值导数。
    从 x0 开始迭代

    这个方法有一些很复杂的地方，涉及到混沌理论，但这里不讨论
    这些复杂的地方...简而言之，牛顿迭代法的收敛性取决于初值如何。

    经典的牛顿迭代法求单根时有二阶收敛性，但求重根时只有线性收敛。

    当所求的为重根 multiple root 时，应该按照重数 multiplicity
    来改良迭代方式。
    '''
    if df == None: df = lambda x: (f(x+(1e-5))-f(x-(1e-5)))*(5e4)
    phi = lambda x0: x0 - f(x0)/df(x0)
    return fpi(phi, x0, stop, showlog)

def Newton_relaxation(
    f:Callable[[X],Y],
    x0:X0=0,
    df:Callable[[X],Y] = None,
    stop:StopCondition = stopAt(),
    showlog: bool = False,
    m:int = 1) -> X:
    '''按照重数 m 设置松弛系数，以改良迭代方式的牛顿迭代法。'''
    if df == None: df = lambda x: (f(x+(1e-5))-f(x-(1e-5)))*(5e4)
    phi = lambda x0: x0 - m*f(x0)/df(x0)
    return fpi(phi, x0, stop, showlog)

def Newton_derivative(
    f:Callable[[X],Y] = None,
    x0:X0 = 0,
    df:Callable[[X],Y] = None,
    mu:Callable[[X],Y] = None,
    dmu:Callable[[X],Y]= None,
    stop:StopCondition = stopAt(),
    showlog: bool = False) -> X:
    '''利用代数原理，求 f(x) 的重根等同于求 f(x)/df(x) 的单根的牛顿迭代法。
    其中 mu(x) = f(x)/df(x), dmu 为 mu 的导数
    特别的，f 和 mu 请至少输入一个。'''
    if mu == None:
        if f == None: raise ValueError("f 和 mu 至少要输入一个")
        if df == None: df = lambda x: (f(x+(1e-5))-f(x-(1e-5)))*(5e4)
        mu = lambda x: f(x)/df(x)
    if dmu== None: dmu= lambda x: (mu(x+(1e-5))-mu(x-(1e-5)))*(5e4)
    phi = lambda x0: x0 - mu(x0)/dmu(x0)
    return fpi(phi, x0, stop, showlog)

def secant(
    f:Callable[[X],Y],
    x0:X0=0,
    x1:X1=1,
    stop:StopCondition = stopAt(),
    showlog: bool = False) -> X:
    '''弦截法（二步法）
    类似牛顿法，但是取 (f(x[k]) - f(x[k-1]))/(x[k] - x[k-1]) 作为数值微分
    该方法有 (math.sqrt(5) + 1)/2 约等于 1.618 阶收敛
    
    和牛顿法相似，都是对 f 进行线性插值，然后按线性插值结果求解
    但牛顿法的插值只考虑一个点的函数值和其斜率
    而弦截法的插值是对两个点进行插值'''
    x  = [x1, x0]
    fx = [f(x1), f(x0)]
    time = 0
    if showlog: print(f"开始迭代，初值为：{x0},{x1}")
    while not stop(x[0], x[-1], time):
        newx = x[0] - fx[0]*(x[0]-x[-1])/(fx[0]-fx[-1])
        x[0],x[-1] = newx,x[0]
        fx[0],fx[-1] = f(newx),fx[0]
        time += 1
        if showlog: print(f"第{time}步迭代结果：{newx}")
    return x[0]

def parabolic(
    f:Callable[[X],Y],
    x0:X0=0,
    x1:X1=1,
    x2:X2=2,
    stop:StopCondition = stopAt(),
    showlog: bool = False) -> X:
    '''抛物线法（三步法）
    此方法使用二次多项式对初始的三个点进行插值，
    然后求二次多项式的根
    
    插值方法使用牛顿差商法

    对于 p**3 - p**2 - p - 1 = 0，该方法 p 约等于 1.840 阶收敛
    
    即便初值都为实数，抛物线法也可以求复根；
    而牛顿法仅在初值为复数或函数为复函数时才能求复根。'''
    x   = [x2, x0, x1]
    fx  = [f(x2), f(x0), f(x1)]
    dfx = [(fx[0]-fx[-1])/(x[0]-x[-1]),
           (fx[-1]-fx[-2])/(x[-1]-x[-2])]
    time = 0

    if showlog: print(f"开始迭代，初值为：{x0},{x1},{x2}")
    while not stop(x[0], x[-1], time):
        ddf = (dfx[0]-dfx[-1])/(x[0]-x[-2])
        omega = dfx[0] + ddf*(x[0]-x[-1])
        newx = x[0] - 2*fx[0]/(omega*(1+math.sqrt(1-4*fx[0]*ddf/omega**2)))

        x[0],x[-1],x[-2] = newx,x[0],x[-1]
        fx[0],fx[-1],fx[-2] = f(newx),fx[0],fx[-1]
        dfx[0],dfx[-1] = (fx[0]-fx[-1])/(x[0]-x[-1]),dfx[0]
        time += 1
        if showlog: print(f"第{time}步迭代结果：{newx}")
    return x[0]

if __name__ == "__main__":
    f = lambda x: x**3-3*x-1
    df= lambda x: 3*x**2-3
    print("牛顿迭代法")
    Newton(f=f, x0=2, df=df, showlog=True)
    print("弦切法")
    secant(f=f, x0=2, x1=1.9, showlog=True)
    print("抛物线法")
    parabolic(f=f, x0=1, x1=3, x2=2, showlog=True)