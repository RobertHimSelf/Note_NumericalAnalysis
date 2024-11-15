#!/usr/bin/python
# -*- coding: utf-8 -*-
from numbers import Number
from typing import Callable, TypeVar
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
import math

'''u_t - v*u_{xx} = 0'''

try: from .ode__typing import X,U,T
except: from ode__typing import X,U,T

#边界条件
DIRICHLET = TypeVar("d")
NEWMANN = TypeVar("n")
CIRCULATION = TypeVar("c") 
#如果border_type == CIRCULATION, 则此处边界之外的数值永远等同于另一侧边界的数值。无论border如何设置。

type BorderCondition = DIRICHLET|NEWMANN|CIRCULATION

def FTCS_Spread(
    u0:Callable[[X],U],
    border_min:Callable[[T],U],
    border_max:Callable[[T],U],
    dx:X, dt:T,
    x_min:X, x_max:X,
    t_max:T, v:Number,
    border_min_type:BorderCondition = DIRICHLET,
    border_max_type:BorderCondition = DIRICHLET
    ):
    '''Forward Time Centered Space
    前向时间-中心差分方法，使用狄利克雷边值条件
    解 u_t - v*u_{xx} = 0 的扩散方程
    
    这好像是一个带特定碰撞项的D1Q3格子玻尔兹曼方法？'''
    #初始处理
    xs = round((x_max-x_min)/dx)+1
    ts = math.ceil(t_max/dt)+1
    x = np.linspace(x_min,x_max,xs)
    t = np.linspace(    0,t_max,ts)
    u = np.zeros((xs, ts), dtype=float)
    
    #代入初值
    for x_index in range(xs):
        u[x_index, 0] = u0(x[x_index])
    
    vdt_dxsqr = v*dt/dx**2 #缓存会使用多次的计算结果
    for t_index in range(1, ts):
        #计算 u(x, t_index) 的值

        #处理左边界
        t_last = t_index-1
        if border_min_type == DIRICHLET:
            u[0, t_index] = border_min(t[t_index])
        elif border_min_type == NEWMANN:
            u[0, t_index] = u[0, t_last] + vdt_dxsqr*(u[1, t_last] - u[0, t_last] - border_min(t[t_index]))
        elif border_min_type == CIRCULATION:
            u[0, t_index] = u[0, t_last] + vdt_dxsqr*(u[1, t_last] - 2*u[0, t_last] + u[-1, t_last])
        else: raise ValueError("边界条件类型不被支持")

        #处理右边界
        if border_max_type == DIRICHLET:
            u[-1,t_index] = border_max(t[t_index])
        elif border_max_type == NEWMANN:
            u[-1, t_index] = u[-1, t_last] + vdt_dxsqr*(u[-2, t_last] - u[-1, t_last] + border_max(t[t_index]))
        elif border_max_type == CIRCULATION:
            u[-1, t_index] = u[0, t_last] + vdt_dxsqr*(u[-2, t_last] - 2*u[-1, t_last] + u[0, t_last])
        else: raise ValueError("边界条件类型不被支持")

        #对非边界的部分进行计算
        u[1:xs-1, t_index] = u[1:xs-1, t_last] + vdt_dxsqr*(
            u[2:, t_last] - 2*u[1:xs-1, t_last] + u[:xs-2, t_last])
    return x,t,u

def BTCS_Spread(
    u0:Callable[[X],U],
    border_min:Callable[[T],U],
    border_max:Callable[[T],U],
    dx:X, dt:T,
    x_min:X, x_max:X,
    t_max:T, v:Number,
    border_min_type=DIRICHLET,
    border_max_type=DIRICHLET
    ):
    '''Backward Time Centered Space
    后向时间-中心差分方法，使用狄利克雷边值条件
    解 u_t - v*u_{xx} = 0 的扩散方程'''
    #初始处理
    xs = round((x_max-x_min)/dx)+1
    ts = math.ceil(t_max/dt)+1
    x = np.linspace(x_min,x_max,xs)
    t = np.linspace(    0,t_max,ts)
    u = np.zeros((xs, ts), dtype=float)

    #代入初值
    for x_index in range(xs):
        u[x_index, 0] = u0(x[x_index])

    #缓存会多次使用的计算结果
    a = -v*dt/dx**2
    temp = 1-2*a

    #各边值条件所生成矩阵的讨论
    #DIRICHLET                  #NEWMANN            #CIRCULATION
    #               [bmin]      #                   #[1-2a a 0 0 a] []
    #[a 1-2a a 0 0] []          #[1-a-bmin a 0] []  #[a 1-2a a 0 0] []
    #[0 a 1-2a a 0] []          #[a   1-2a   a] []  #[0 a 1-2a a 0] []
    #[0 0 a 1-2a a] []          #[0 a 1-a+bmax] []  #[0 0 a 1-2a a] []
    #               [bmax]      #                   #[a 0 0 a 1-2a] []
    
    #不难发现，循环边值条件有无穷多个解。需要额外讨论。
    #即便在只有一边为循环边值条件时，得到的也并非简单三对角矩阵。因此这里不再涉及。
    if border_max_type == CIRCULATION or border_min_type == CIRCULATION:
        raise ValueError("后向时间-中心差分方法不支持循环边值条件")
    
    #如果两边均为 DIRICHLET 边界，则生成的矩阵实际上是仅由 a, 1-2a 组成的、最简单的三对角矩阵。
    #这里暂时仅考虑两边均为 DIRICHLET 边界的情况。
    easy_mode = False
    if border_min_type == DIRICHLET and border_max_type == DIRICHLET:
        easy_mode = True
        #三对角矩阵可以分离成“下三角三对角矩阵”和“上三角三对角矩阵”的乘积。从而可以用特殊解法。
        #详情参见 le_direct_specialmat.py
        alp = [temp]
        bet = []
        for _ in range(xs-3):
            bet.append(a/alp[-1])
            alp.append(temp-a*bet[-1])
    else:
        #将来可能会考虑 NEWMANN 边界的情况。
        raise Exception("NOT SUPPORT YET")

    for t_index in range(1, ts):
        t_last = t_index-1
        if easy_mode:
            u[0, t_index] = border_min(t[t_index])
            u[-1,t_index] = border_max(t[t_index])
            n = len(alp)#=xs-2
            u[1, t_index] = u[1, t_last]/alp[0]
            for x_index in range(2,xs-1):
                u[x_index, t_index] = (u[x_index, t_last]-u[x_index-1, t_index]*a)/alp[x_index-1]
            for x_index in range(xs-3, 0, -1):
                u[x_index, t_index] -= bet[x_index-1]*u[x_index+1, t_index]
    return x,t,u

if __name__ == "__main__":
    pass