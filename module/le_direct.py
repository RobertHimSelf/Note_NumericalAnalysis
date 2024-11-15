#!/usr/bin/python
# -*- coding: utf-8 -*-

'''解线性方程组的直接法
这些方法都是 o(n^3) 的'''

import numpy as np

try: from ._matfunc import *
except: from _matfunc import *

#a*x = b

def Gauss_origin(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    '''经典高斯消去法，不进行任何高级操作'''
    _a = a.copy()
    _b = b.copy()
    n = _a.shape[0]
    for i in range(n):            #行变换形成上三角矩阵
        if _a[i,i] == 0:
            print("出错，计算过程中出现主元为零：")
            print(_a)
            raise ValueError("主元为零")
        for j in range(i+1, n):
            _b[j,:] -= _b[i,:]*(temp:=_a[j,i]/_a[i,i])
            _a[j,:] -= _a[i,:]*(temp)
    for i in range(n-1, -1, -1):  #行变换形成单位矩阵
        _b[i,:] /= _a[i,i]
        _a[i,i] = 1
        for j in range(i):
            _b[j,:] -= _b[i,:]*(_a[j,i])
            _a[j,:] -= _a[i,:]*(_a[j,i])
    return _b

type L = np.ndarray
type U = np.ndarray

def lu_origin(a:np.ndarray) -> tuple[L,U]: #TODO DEBUG
    '''LU分解法，a = l*u'''
    u = a.copy()
    n = a.shape[0]
    l = np.ndarray([[0,]*i+[1,]+[0,]*(n-i-1) for i in range(n)])
    for i in range(n):            #行变换形成上三角矩阵
        if u[i,i] == 0:
            print("出错，计算过程中出现主元为零：")
            print(l)
            raise ValueError("主元为零")
        for j in range(i+1, n):
            l[j,:] -= l[i,:]*(temp:=u[j,i]/u[i,i])
            u[j,:] -= u[i,:]*(temp)
    return l, u

def lu_origin_SubstitudeBack(lu:tuple[L,U],b:np.ndarray):
    '''LU分解法回代，l*u*x=b'''
    _b = b.copy()
    l,u = lu
    n = l.shape[0]
    c = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i):
            b[i,0] -= c[j]*l[i,j]
        c[i]=b[i,0]
    x = np.ndarray(np.zeros((1,n), dtype=float))
    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            c[i]-=u[i,j]*x[j,0]
        x[i,0]=c[i]/u[i,i]
    return x

type LU = np.ndarray

def lu_memorysave(a:np.ndarray) -> LU:
    '''LU分解法，a = l*u
    注意到有效的内容都在 l 的下三角区域（不含对角线）和 u 的上三角区域（含对角线）。
    所以两者可以储存在同一个矩阵中。从而节省一半内存。'''
    lu = a.copy()
    n = a.shape[0]
    for i in range(n):            #行变换形成上三角矩阵
        if lu[i,i] == 0:
            print("出错，计算过程中出现主元为零：")
            print(lu)
            raise ValueError("主元为零")
        for j in range(i+1, n):
            lu[j,i] /= -lu[i,i]
            lu[j,:i] += lu[i,:i]*lu[j,i]
            lu[j,i+1:] -= lu[i,i+1:]*lu[j,i]
    return lu

def lu_memorysave_SubstitudeBack(lu:LU,b:np.ndarray):
    '''LU分解法回代，l*u*x=b
    对应节省内存的lu分解法'''
    n = lu.shape[0]
    c = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i):
            b[i,0] -= c[j]*lu[i,j]
        c[i]=b[i,0]
    x = np.ndarray(np.zeros((1,n), dtype=float))
    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            c[i]-=lu[i,j]*x[j,0]
        x[i,0]=c[i]/lu[i,i]
    return x

def Gauss(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    '''部分主元消去法，通过行对调将该列绝对值最大的元素转化为主元
    p*a*x=p*b, p 为交换矩阵'''
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
        for j in range(i+1, n):
            _b[j,:] -= _b[i,:]*(temp:=_a[j,i]/_a[i,i])
            _a[j,:] -= _a[i,:]*(temp)
    for i in range(n-1, -1, -1):  #行变换形成单位矩阵
        _b[i,:] /= _a[i,i]
        _a[i,i] = 1
        for j in range(i):
            _b[j,:] -= _b[i,:]*(_a[j,i])
            _a[j,:] -= _a[i,:]*(_a[j,i])
    return _b

type ARRANGE = list[int]

def lu(a:np.ndarray) -> tuple[LU, ARRANGE]:
    '''LU分解法，p*a = l*u
    注意到有效的内容都在 l 的下三角区域（不含对角线）和 u 的上三角区域（含对角线）。
    所以两者可以储存在同一个矩阵中。从而节省一半内存。'''
    lu = a.copy()
    arrange = [i for i in range(a.shape[0])]
    n = a.shape[0]
    for i in range(n):            #行变换形成上三角矩阵
        row = i + np.argmax(np.abs(lu[i:,i])) #找到绝对值最大元素所在行
        swaprow(lu,row,i)   #进行对调
        arrange[row],arrange[i]=arrange[i],arrange[row]
        if lu[i,i] == 0:
            print("出错，计算过程中出现主元为零：")
            print(lu)
            raise ValueError("主元为零")
        for j in range(i+1, n):
            lu[j,i] /= -lu[i,i]
            lu[j,:i] += lu[i,:i]*lu[j,i]
            lu[j,i+1:] -= lu[i,i+1:]*lu[j,i]
    return lu, arrange

def lu_memorysave_SubstitudeBack(lu:tuple[LU, ARRANGE],b:np.ndarray):
    '''LU分解法回代，l*u*x=p*b
    对应节省内存的lu分解法'''
    n = lu[0].shape[0]
    arrangerow(b, lu[1])
    c = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i):
            b[i,0] -= c[j]*lu[0][i,j]
        c[i]=b[i,0]
    x = np.ndarray(np.zeros((1,n), dtype=float))
    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            c[i]-=lu[0][i,j]*x[j,0]
        x[i,0]=c[i]/lu[0][i,i]
    return x

def GaussJordan(a:np.ndarray) -> np.ndarray:
    '''求逆矩阵 a*r = i, 给出 r
    回代时，有 a*r*b = b 从而 r*b 为解
    具体到这个库，要 np.matmul(result, b)'''
    b = np.zeros(a.shape)
    for i in range(a.shape[0]):
        b[i,i]=1
    return Gauss_origin(a, b)

def GaussJordanP(a:np.ndarray) -> tuple[np.ndarray,ARRANGE]:
    '''求逆矩阵 p*a*r = i, 给出 r
    回代时，有 p*a*r*b = p*b 从而 r*p*b 为解
    具体到这个库，要 np.matmul(result[0], arrangerow(b,result[1]))'''
    _b = np.zeros(a.shape)
    for i in range(a.shape[0]):
        _b[i,i]=1
    _a = a.copy()
    n = _a.shape[0]
    arrange = [i for i in range(n)]
    for i in range(n):
        row = i + np.argmax(np.abs(_a[i:,i])) #找到绝对值最大元素所在行
        swaprow(_a,row,i)   #进行对调
        swaprow(_b,row,i)
        arrange[row],arrange[i]=arrange[i],arrange[row]
        if _a[i,i] == 0:
            print("出错，计算过程中出现主元为零：")
            print(_a)
            raise ValueError("主元为零")
        for j in range(i+1, n):
            _b[j,:] -= _b[i,:]*(temp:=_a[j,i]/_a[i,i])
            _a[j,:] -= _a[i,:]*(temp)
    for i in range(n-1, -1, -1):  #行变换形成单位矩阵
        _b[i,:] /= _a[i,i]
        _a[i,i] = 1
        for j in range(i):
            _b[j,:] -= _b[i,:]*(_a[j,i])
            _a[j,:] -= _a[i,:]*(_a[j,i])
    return _b, arrange