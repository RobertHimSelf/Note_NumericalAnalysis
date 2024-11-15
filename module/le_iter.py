#!/usr/bin/python
# -*- coding: utf-8 -*-

'''解线性方程组的迭代法
若迭代次数为 k，则这些方法都是 o(k*n^2)的'''

import numpy as np
from numbers import Number

try:
    from ._matfunc import *
    from .iter_condition import StopCondition, astopAt
except:
    from _matfunc import *
    from iter_condition import StopCondition, astopAt


#a*x = b

def Jacobi(
    a:np.ndarray, b:np.ndarray, 
    x0:np.ndarray = None,
    stop:StopCondition = astopAt()) -> np.ndarray:
    '''雅可比迭代法
    令a=l+d+u，进行 d*new_x = b-(l+u)*x 迭代。
    在主对角优势矩阵的线性方程组中，雅可比迭代法收敛。但对角优势并非收敛的必要条件。
    '''
    _a = a.copy()
    n = _a.shape[0]
    if x0 == None: x0 = b.copy()

    d = []  #分离 d 和 l+u
    for i in range(n):
        d.append(1/_a[i,i])
        _a[i,i] = 0
    
    time = 0
    x = np.zeros((n,1))
    while not stop(x, x0, time):
        x[:] = x0
        x0 = mul_perrow(b-np.matmul(_a, x), d)
        time += 1
    return x0

def GaussSeidel(
    a:np.ndarray, b:np.ndarray, 
    x0:np.ndarray = None,
    stop:StopCondition = astopAt()) -> np.ndarray:
    '''高斯-赛德尔迭代法
    令a=l+d+u，进行 (l+d)*new_x = b-u*x 迭代。
    在主对角优势矩阵的线性方程组中，高斯-赛德尔迭代法收敛。但对角优势并非收敛的必要条件。
    收敛速度一般比雅可比迭代法快，但稳定性比雅可比迭代法略差。
    由于阶梯矩阵运算性质，实际上用的是 d*new_x = b-l*new_x-u*x'''
    _a = a.copy()
    n = _a.shape[0]
    if x0 == None: x0 = b.copy()
    time = 0
    x = np.zeros((n,1))
    d = []  #分离 d 和 l+u
    for i in range(n):
        d.append(1/_a[i,i])
        _a[i,i] = 0

    while not stop(x, x0, time):
        x[:] = x0
        for i in range(n-1,-1,-1):
            x0[i,0] = d[i]*(_a[i,:]*x0[:,0]).sum()
    return x0

def successive_relaxation(
    a:np.ndarray, b:np.ndarray, 
    x0:np.ndarray = None,
    stop:StopCondition = astopAt(),
    alpha:Number = 1) -> np.ndarray:
    '''逐次松弛迭代法
    记高斯赛德尔迭代法每一步移动的向量为 dx，则该方法每一步移动 alpha*dx。

    alpha 即为松弛参数。相比高斯-赛德尔迭代法而言，该方法有以下特征：
    当 alpha > 1，方法为逐次超松弛迭代法SOR，数值稳定性差，但收敛速度更快；
    当 alpha == 1，方法就是高斯赛德尔迭代法；
    当 0 < alpha < 1，方法为逐次次松弛迭代法，数值稳定性更好，但收敛速度更慢；
    当 alpha <= 0，方法不收敛。
    
    雅可比迭代法也可以松弛，但这里不再涉及了。'''
    _a = a.copy()
    n = _a.shape[0]
    if x0 == None: x0 = b.copy()
    time = 0
    x = np.zeros((n,1))

    d = []  #分离 d 和 l+u
    for i in range(n):
        d.append(1/_a[i,i])
        _a[i,i] = 0

    while not stop(x, x0, time):
        x[:,0] = x0
        for i in range(n-1,-1,-1):
            x0[i,0] = d[i]*(_a[i,:]*x0[:,0]).sum()
        x0[:] = x + alpha*(x0 - x)
    return x0
