#!/usr/bin/python
# -*- coding: utf-8 -*-

'''预条件（Precondition）
#通过使用预条件技术，可以使迭代法收敛速度加快。迭代法收敛率通常直接或间接依赖于系数矩阵A的条件数，预条件方法就是降低矩阵A的条件数的。
#预条件的基本形式是M^(-1)Ax=M^(-1)b。其中M为可逆矩阵，被称为预条件子。
'''

import numpy as np
from numbers import Number

try:
    from ._matfunc import *
    from .iter_condition import StopCondition, astopAt
except:
    from _matfunc import *
    from iter_condition import StopCondition, astopAt


#a*x = b
def Jacobi_Precondition(
    a:np.ndarray, b:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''用雅可比预条件子处理 a 和 b'''
    _a = a.copy()
    _b = b.copy()
    n = _a.shape[0]
    for i in range(n):
        if a[i,i] == 0:
            raise ValueError("主对角线元素为零")
        temp = 1/a[i,i]
        for j in range(i):
            _a[i,j] *= temp
        for j in range(i+1,n):
            _a[i,j] *= temp
        _b[i,0] *= temp
        _a[i,i] = 1
    return _a,_b

def Jacobi_Precondition_Advanced(
    a:np.ndarray, b:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''用雅可比预条件子处理 a 和 b。增加行对调。'''
    _a = a.copy()
    _b = b.copy()
    n = _a.shape[0]
    for i in range(n):
        row = i + np.argmax(np.abs(_a[i:,i])) #找到绝对值最大元素所在行
        swaprow(_a,row,i)   #进行对调
        swaprow(_b,row,i)
        if _a[i,i] == 0:
            raise ValueError("主对角线元素为零")
        temp = 1/_a[i,i]
        for j in range(i):
            _a[i,j] *= temp
        for j in range(i+1,n):
            _a[i,j] *= temp
        _b[i,0] *= temp
        _a[i,i] = 1
    return _a,_b

def Jacobi_With_Precondition(
    a:np.ndarray, b:np.ndarray, 
    x0:np.ndarray = None,
    stop:StopCondition = astopAt()) -> np.ndarray:
    '''雅可比迭代法，使用预条件子，显著提高迭代速度'''
    _a = a.copy()
    _b = b.copy()
    n = _a.shape[0]
    for i in range(n):
        row = i + np.argmax(np.abs(_a[i:,i])) #找到绝对值最大元素所在行
        swaprow(_a,row,i)   #进行对调
        swaprow(_b,row,i)
        if _a[i,i] == 0:
            print("出错，计算过程中出现主元为零：")
            print(_a)
            raise ValueError("主元为零")
        temp = 1/_a[i,i]
        for j in range(i):
            _a[i,j] *= temp
        for j in range(i+1,n):
            _a[i,j] *= temp
        _b[i,0] *= temp
        _a[i,i] = 0

    if x0 == None: x0 = b.copy()
    
    time = 0
    x = np.zeros((n,1))
    while not stop(x, x0, time):
        x[:] = x0
        x0[:] = _b-np.matmul(_a, x)
        time += 1
    return x0