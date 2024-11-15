#!/usr/bin/python
# -*- coding: utf-8 -*-

'''解特殊线性方程组的直接法'''

from numbers import Number

#三对角矩阵
# b_0 c_0
# a_0 b_1 c_1
#     a_1 b_2 c_2
#         a_2 b_3
type Diagnal3 = tuple[list[Number],list[Number],list[Number]]|list[list[Number]]
#以 (a, b, c) 的方式存储，其中 a = [a_0, a_1, ...], ...

def d3(d3mat: Diagnal3):
    #讲 a 分解为 (l+d)*(i+u)，其中 l 和 u 仅次对角线非零，用列表表示
    l,b,c = d3mat[0],d3mat[1],d3mat[2]
    d = list()
    u = list()
    d.append(b[0])
    for i in range(len(l)):
        u.append(c[i]/d[-1])
        d.append(b[i+1]-l[i]*u[-1])
    return l,d,u

def d3_SubstitudeBack(d3, b:list[Number]) -> list[Number]:
    gam, alp, bet = d3
    n = len(alp)
    x = [b[0]/alp[0]]
    for i in range(1,n):
        x.append((b[i]-x[-1]*gam[i-1])/alp[i])
    for i in range(n-2, -1, -1):
        x[i] -= bet[i]*x[i+1]
    return x

if __name__ == "__main__":
    import numpy as np
    l,d,u = d3([[1,2,3],[4,5,6,7],[1,2,3]])
    left = np.array(
       [[d[0], 0, 0, 0],
        [l[0],d[1],0,0],
        [0,l[1],d[2],0],
        [0,0,l[2],d[3]]])
    right= np.array(
       [[1,u[0],0,0],
        [0,1,u[1],0],
        [0,0,1,u[2]],
        [0,0,0,1]]
    )
    print(np.matmul(left,right))