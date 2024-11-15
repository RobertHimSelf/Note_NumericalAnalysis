#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


a = np.array([[1,0],[0,1]])
#获得列向量
print(a[:,0])
#获得行向量
print(a[0,:])
#获得子矩阵
print(a[:1,0])
#获得元素
print(a[0,0])

#列变换
a[0,:] += a[1,:]
print(a)
#行变换类似，子矩阵变换类似


b = np.array([[1,0],[0,1],[1,1]])
#获得行列
print(b.shape)

#列对调
a[0,:],a[1,:]=a[1,:].copy(),a[0,:].copy()

