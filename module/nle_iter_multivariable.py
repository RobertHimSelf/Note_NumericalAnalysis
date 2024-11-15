#!/usr/bin/python
# -*- coding: utf-8 -*-

'''解方程组的不动点迭代法'''

import numpy as np
from numbers import Number
from typing import Callable

try:
    from ._matfunc import matmul
    from .iter_condition import StopCondition, astopAt
    from .le_direct import Gauss
except:
    from _matfunc import matmul
    from iter_condition import StopCondition, astopAt
    from le_direct import Gauss

#类似解方程的不动点迭代法，解方程组同样也有不动点迭代法。
#这是不动点迭代法的基础形式，与之前方法的区别只在于类型标注不同
#以及 stop 的默认值变成应用于 array 的 astopAt
def afpi(
    phi: Callable[[np.ndarray],np.ndarray],
    x0: np.ndarray,
    stop: StopCondition = astopAt(),
    showlog: bool = False) -> np.ndarray:
    '''Fixed point iteration 不动点迭代法'''
    x = x0; time = 0
    if showlog: print(f"开始迭代，初值为：{x.flatten()}")
    while not stop(x, x0, time):
        x0 = x
        x = phi(x0)
        time += 1
        if showlog: print(f"第{time}步迭代结果：{x.flatten()}")
    return x

#对向量的迭代进行收敛性讨论时，要使用度量（范数）与压缩映射原理
#类似，可以定义收敛性、p阶收敛等。

def aNewton(
    f:Callable[[np.ndarray], np.ndarray],
    x0:np.ndarray,
    df:Callable[[np.ndarray], np.ndarray] = None,
    lesolver:Callable[[np.ndarray, np.ndarray], np.ndarray] = Gauss,
    stop:StopCondition = astopAt(),
    showlog: bool = False) -> np.ndarray:
    '''解 f(x) = 0 的牛顿迭代法
    其中 x 为向量，f(x) 同样为一个向量。
    df(x) 为 f 在 x 处的雅可比矩阵，如果 df 为 None，则使用默认的数值导数。
    从 x0 开始迭代'''
    n = x0.shape[0]
    if df == None:
        def jacobi_matrix(x):
            y = f(x)
            matrix = np.zeros((n,n))
            for i in range(n):
                deltax = np.zeros(n)
                deltax[i] += 1e-5
                matrix[i,:] = (f(x + deltax) - y)*(1e5)
            return matrix
        df = jacobi_matrix
    phi = lambda x: x - lesolver(df(x), f(x))
    return afpi(phi, x0, stop, showlog)

def Broyden(
    f:Callable[[np.ndarray], np.ndarray],
    x0:np.ndarray,
    df0:np.ndarray = None,
    lesolver:Callable[[np.ndarray, np.ndarray], np.ndarray] = Gauss,
    stop:StopCondition = astopAt(),
    showlog:bool = False) -> np.ndarray:
    '''解 f(x) = 0 的 Broyden 方法
    其中 x 为向量，f(x) 同样为一个向量。
    df0 为 f 在 x0 处的雅可比矩阵近似值，如果没有更好的近似值，可以使用单位矩阵。
    本方法在迭代过程中，会逐步求出近似的雅可比矩阵。
    从 x0 开始迭代'''
    x = x0; time = 0
    n = x0.shape[0]
    if df0 == None:
        df0 = np.array([[
            (1 if i==j else 0)
            for i in range(n)] for j in range(n)],
            dtype=float)
    df = df0
    f_list:list[np.ndarray] = [f(x), 0] #[f(x), f(x0)]
    delta_list:list[np.ndarray] = [0, 0]

    if showlog: print(f"开始迭代，初值为：{x.flatten()}")
    while True:
        x0 = x
        delta_list[0] = lesolver(df,f_list[0])
        x = x0 - delta_list[0]
        time += 1
        if showlog: print(f"第{time}步迭代结果：{x.flatten()}")

        if stop(x, x0, time): break
        f_list[1] = f_list[0]
        f_list[0] = f(x)
        df0 = df
        df = df0 + matmul(
            f_list[0]-f_list[1]-matmul(df, delta_list[1].reshape((n,1))),
            delta_list[0].reshape((1,n)))/np.dot(delta_list[0], delta_list[0])
        delta_list[1] = delta_list[0]
    return x

def BroydenII(
    f:Callable[[np.ndarray], np.ndarray],
    x0:np.ndarray,
    b0:np.ndarray = None,
    stop:StopCondition = astopAt(),
    showlog:bool = False) -> np.ndarray:
    '''解 f(x) = 0 的 Broyden 二号方法
    其中 x 为向量，f(x) 同样为一个向量。
    b 为 f 在 x0 处的雅可比矩阵的*逆矩阵*的近似值，如果没有更好的近似值，可以使用单位矩阵。
    本方法在迭代过程中，会逐步求出雅可比矩阵的逆的近似值。
    从 x0 开始迭代'''
    x = x0; time = 0
    n = x0.shape[0]
    if b0 == None:
        b0 = np.array([[
            (1 if i==j else 0)
            for i in range(n)] for j in range(n)],
            dtype=float)
    b = b0
    fx = f(x)

    if showlog: print(f"开始迭代，初值为：{x.flatten()}")
    while True:
        x0 = x; fx0 = fx
        dx = -matmul(b, fx0.reshape((n,1)))
        x = x0 - dx
        time += 1
        if showlog: print(f"第{time}步迭代结果：{x.flatten()}")

        if stop(x0, x, time): break
        fx = f(x)
        dfx= fx-fx0
        b = b + matmul(dx - matmul(b,dfx.reshape((n,1))), dx.reshape((1,n)), b)/matmul(dx.reshape((1,n)),b,dfx.reshape(n,1))
    return x

if __name__ == "__main__":
    def f(x):
        return np.array(
            [[x[0,0]**2 + x[1,0]**2 - 4], 
             [x[0,0]**2 - x[1,0]**2 - 1]])
    def df(x):
        return np.array(
            [[2*x[0,0], 2*x[1,0]],
            [2*x[0,0], -2*x[1,0]]]
        )
    x0=np.array([[1.6,],[1.2,]])
    print("牛顿迭代法")
    
    aNewton(f=f, x0=x0, df=df, showlog=True)